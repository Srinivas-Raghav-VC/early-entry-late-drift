from __future__ import annotations

from typing import Dict, List, Optional

import torch

from core import get_model_layers  # noqa: E402


class HookGroup:
    def __init__(self, handles):
        self.handles = list(handles)

    def remove(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass


def infer_num_heads(model, attn_module) -> int:
    candidates = [
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(getattr(attn_module, "config", None), "num_attention_heads", None),
        getattr(getattr(model, "config", None), "num_attention_heads", None),
        getattr(getattr(getattr(model, "config", None), "text_config", None), "num_attention_heads", None),
    ]
    for candidate in candidates:
        try:
            value = int(candidate)
        except Exception:
            value = 0
        if value > 0:
            return value
    raise RuntimeError("Could not infer num_attention_heads")


def get_attn_module(model, layer: int):
    layers = get_model_layers(model)
    attn_module = getattr(layers[int(layer)], "self_attn", None)
    if attn_module is None:
        attn_module = getattr(layers[int(layer)], "self_attention", None)
    if attn_module is None or not hasattr(attn_module, "o_proj"):
        raise AttributeError(f"Layer {layer} attention has no o_proj")
    return attn_module


def capture_o_proj_inputs_at_position(model, input_ids: torch.Tensor, layers_to_capture: List[int], pos: int) -> Dict[int, torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    for layer in layers_to_capture:
        attn_module = get_attn_module(model, int(layer))

        def make_hook(layer_idx: int):
            def pre_hook(module, inputs_tuple):
                if not inputs_tuple:
                    return None
                x = inputs_tuple[0]
                if not torch.is_tensor(x) or x.ndim != 3:
                    return None
                if int(pos) >= int(x.shape[1]):
                    return None
                captured[int(layer_idx)] = x[0, int(pos), :].detach().clone()
                return None
            return pre_hook

        handles.append(attn_module.o_proj.register_forward_pre_hook(make_hook(int(layer))))

    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [int(layer) for layer in layers_to_capture if int(layer) not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture o_proj inputs for layers={missing}")
    return captured


def register_attention_head_replace_hooks(
    model,
    donor_vectors_by_layer: Dict[int, torch.Tensor],
    group_layers: Dict[int, List[int]],
    *,
    replace_position: int,
):
    handles = []
    for layer, heads in sorted(group_layers.items()):
        attn_module = get_attn_module(model, int(layer))
        num_heads = infer_num_heads(model, attn_module)
        donor_vec = donor_vectors_by_layer[int(layer)].detach().clone()
        if int(donor_vec.numel()) % int(num_heads) != 0:
            raise RuntimeError(f"Layer {layer}: donor vector shape incompatible with num_heads")
        head_dim = int(donor_vec.numel()) // int(num_heads)
        donor_heads = donor_vec.view(int(num_heads), int(head_dim))
        chosen_heads = [int(head) for head in heads]

        def make_hook(clean_heads: torch.Tensor, chosen: List[int], n_heads: int, pos: int):
            def pre_hook(module, inputs_tuple):
                if not inputs_tuple:
                    return None
                x = inputs_tuple[0]
                if not torch.is_tensor(x) or x.ndim != 3:
                    return None
                if int(pos) >= int(x.shape[1]):
                    return None
                d_model = int(x.shape[2])
                if d_model % int(n_heads) != 0:
                    return None
                head_dim_local = d_model // int(n_heads)
                x_new = x.clone()
                row = x_new[0, int(pos), :].view(int(n_heads), int(head_dim_local))
                src = clean_heads.to(device=row.device, dtype=row.dtype)
                for head_idx in chosen:
                    if 0 <= int(head_idx) < int(n_heads):
                        row[int(head_idx), :] = src[int(head_idx), :]
                x_new[0, int(pos), :] = row.view(-1)
                if len(inputs_tuple) == 1:
                    return (x_new,)
                return (x_new,) + tuple(inputs_tuple[1:])
            return pre_hook

        handles.append(
            attn_module.o_proj.register_forward_pre_hook(
                make_hook(donor_heads, chosen_heads, num_heads, int(replace_position))
            )
        )
    return HookGroup(handles)


def register_attention_head_addition_hooks(
    model,
    delta_vectors_by_layer: Dict[int, torch.Tensor],
    group_layers: Dict[int, List[int]],
    *,
    replace_position: int,
):
    handles = []
    for layer, heads in sorted(group_layers.items()):
        attn_module = get_attn_module(model, int(layer))
        num_heads = infer_num_heads(model, attn_module)
        delta_vec = delta_vectors_by_layer[int(layer)].detach().clone()
        if int(delta_vec.numel()) % int(num_heads) != 0:
            raise RuntimeError(f"Layer {layer}: delta vector shape incompatible with num_heads")
        head_dim = int(delta_vec.numel()) // int(num_heads)
        delta_heads = delta_vec.view(int(num_heads), int(head_dim))
        chosen_heads = [int(head) for head in heads]

        def make_hook(patch_heads: torch.Tensor, chosen: List[int], n_heads: int, pos: int):
            def pre_hook(module, inputs_tuple):
                if not inputs_tuple:
                    return None
                x = inputs_tuple[0]
                if not torch.is_tensor(x) or x.ndim != 3:
                    return None
                if int(pos) >= int(x.shape[1]):
                    return None
                d_model = int(x.shape[2])
                if d_model % int(n_heads) != 0:
                    return None
                head_dim_local = d_model // int(n_heads)
                x_new = x.clone()
                row = x_new[0, int(pos), :].view(int(n_heads), int(head_dim_local))
                delta = patch_heads.to(device=row.device, dtype=row.dtype)
                for head_idx in chosen:
                    if 0 <= int(head_idx) < int(n_heads):
                        row[int(head_idx), :] = row[int(head_idx), :] + delta[int(head_idx), :]
                x_new[0, int(pos), :] = row.view(-1)
                if len(inputs_tuple) == 1:
                    return (x_new,)
                return (x_new,) + tuple(inputs_tuple[1:])
            return pre_hook

        handles.append(
            attn_module.o_proj.register_forward_pre_hook(
                make_hook(delta_heads, chosen_heads, num_heads, int(replace_position))
            )
        )
    return HookGroup(handles)


def register_keep_only_transcoder_features_hook(
    model,
    transcoder,
    layer: int,
    keep_idx: torch.Tensor,
    *,
    keep_position: Optional[int] = None,
):
    layers = get_model_layers(model)
    keep_idx = keep_idx.detach().to(dtype=torch.long)

    def hook(module, inputs_tuple, output):
        mlp_in = inputs_tuple[0]
        y = output[0] if isinstance(output, tuple) else output

        seq_len = int(mlp_in.shape[1])
        if keep_position is not None:
            if keep_position >= seq_len:
                return output
            pos = int(keep_position)
        else:
            pos = seq_len - 1

        x_pos = mlp_in[:, pos, :]
        y_pos = y[:, pos, :]
        x_pos_f = x_pos.float()
        y_pos_f = y_pos.float()
        cur_feats = transcoder.encode(x_pos_f)
        cur_contrib = transcoder.decode(cur_feats).float()
        residual = y_pos_f - cur_contrib

        kept_feats = torch.zeros_like(cur_feats)
        idx = keep_idx.to(device=cur_feats.device)
        if idx.numel() > 0:
            kept_feats[:, idx] = cur_feats[:, idx]
        new_contrib = transcoder.decode(kept_feats).float()

        y_new = y.clone()
        y_new[:, pos, :] = (residual + new_contrib).to(dtype=y.dtype)
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return layers[layer].mlp.register_forward_hook(hook)
