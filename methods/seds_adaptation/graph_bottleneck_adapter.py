"""C23: zero-initialized bottleneck adapters inside the frozen pose GCN."""
from torch import nn


class GraphBottleneckAdapter(nn.Module):
    """A small residual correction for an ST-GCN block output (N,C,T,V)."""

    def __init__(self, channels, rank=16):
        super().__init__()
        if channels < 1 or rank < 1 or rank > channels:
            raise ValueError('Invalid graph-adapter dimensions')
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.input = nn.Conv2d(channels, rank, 1)
        self.activation = nn.GELU()
        self.output = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, value):
        return value + self.output(self.activation(self.input(self.norm(value))))


def _block_channels(block):
    convolution = block.tcn[3]
    if not isinstance(convolution, nn.Conv2d):
        raise ValueError('Unexpected native ST-GCN block layout')
    return convolution.out_channels


def attach_graph_bottleneck_adapters(model, rank=16):
    """Attach one registered residual adapter after each native graph block.

    The hand stack is shared by left/right hands exactly as in native SEDS; the
    body stack remains separate. Forward hooks alter only the feature tensor and
    return the native adjacency object unchanged.
    """
    if hasattr(model, 'graph_bottleneck_adapters'):
        raise ValueError('Graph bottleneck adapters already attached')
    adapters = nn.ModuleDict()
    handles = []
    prototype = next(model.parameters())
    for stream in ('hand', 'body'):
        backbone = getattr(model.signbert.embed, f'st_gcn_{stream}')
        blocks = [*backbone.st_gcn_networks, *backbone.st_gcn_pool]
        if len(blocks) != 5:
            raise ValueError('Expected three native and two pooled ST-GCN blocks')
        for index, block in enumerate(blocks):
            name = f'{stream}_{index}'
            adapter = GraphBottleneckAdapter(_block_channels(block), rank).to(
                device=prototype.device, dtype=prototype.dtype)
            adapters[name] = adapter

            def adapt(_module, _inputs, result, adapter_name=name):
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError('Expected native ST-GCN (features, adjacency) output')
                features, adjacency = result
                return adapters[adapter_name](features), adjacency

            handles.append(block.register_forward_hook(adapt))
    model.add_module('graph_bottleneck_adapters', adapters)
    # Keep handles alive without registering duplicate modules or state.
    model.__dict__['_graph_bottleneck_adapter_handles'] = handles
    return adapters


def configure_graph_bottleneck_adapters(model, active):
    """Freeze the native pose encoder; train only adapters plus native fusion."""
    for parameter in model.signbert.embed.parameters():
        parameter.requires_grad_(False)
    model.signbert.embed.eval()
    model.graph_bottleneck_adapters.requires_grad_(True)
    model.graph_bottleneck_adapters.train()
    active.append(model.graph_bottleneck_adapters)
    return active


def graph_adapter_learning_rates(optimizer, model, adapter_lr=1e-4, fusion_lr=1e-5):
    if optimizer.state:
        raise ValueError('Fresh optimizer required')
    if adapter_lr != 1e-4 or fusion_lr != 1e-5:
        raise ValueError('Unregistered graph-adapter learning rate')
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    rates = {'fusion': fusion_lr, 'graph_adapter': adapter_lr}
    groups = []
    for group in optimizer.param_groups:
        partitions = {kind: [] for kind in rates}
        for parameter in group['params']:
            if not parameter.requires_grad:
                continue
            name = names[id(parameter)]
            kind = ('fusion' if name.startswith('fusion.') else
                    'graph_adapter' if name.startswith('graph_bottleneck_adapters.') else None)
            if kind is None:
                raise ValueError(f'Unexpected trainable tensor {name}')
            partitions[kind].append(parameter)
        for kind, parameters in partitions.items():
            if parameters:
                groups.append(dict(group, params=parameters, lr=rates[kind],
                                   adaptation_group=kind))
    if {group['adaptation_group'] for group in groups} != set(rates):
        raise ValueError('Missing graph-adapter optimizer group')
    optimizer.param_groups[:] = groups
    return [dict(adaptation_group=group['adaptation_group'], lr=group['lr'],
                 parameters=sum(p.numel() for p in group['params'])) for group in groups]
