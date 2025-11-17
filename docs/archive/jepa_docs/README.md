# Archived JEPA Documentation

These documents were created for the original JEPA implementation and have been superseded by the LeJEPA architecture.

## Archived Files

- `JEPA_ADAPTATION_GUIDE.md` - Guide for adapting JEPA to the codebase (superseded by MPNet-LeJEPA)
- `JEPA_TRAINING_STRATEGY.md` - Training strategies for JEPA (superseded by LeJEPA training)
- `JEPA_EMBEDDING_NETWORK_RAG.md` - JEPA embedding network for RAG (concepts still relevant)

## Current Documentation

See:
- `docs/MPNET_LEJEPA_ARCHITECTURE.md` - Current LeJEPA architecture documentation
- `docs/LEJEPA_IMPLEMENTATION_COMPARISON.md` - Comparison with official LeJEPA

## Migration Notes

The key differences:
- **Old**: JEPA-style predictive loss only
- **New**: LeJEPA = Predictive loss + SIGReg isotropy loss (Epps-Pulley + slicing)
- **Old**: Single encoder or trainable encoder
- **New**: Dual encoders (online + target) with frozen base by default
- **Old**: Optional predictor
- **New**: Always has predictor (online only)

