# Shared research infrastructure

`slr_common` contains method-neutral sign-language retrieval infrastructure: dataset manifests and
tokenization, feature extraction, upstream model bridges, gallery evaluation, transfer utilities,
resource guards, and hashing/provenance primitives. Method packages may depend on this layer;
shared code must not import a method package.
