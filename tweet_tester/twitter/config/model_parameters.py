def setup_params(matrix, tokenizer):
    return {
        'embedding': {
            'OUTPUT_DIM': matrix.EMBEDDING_VECTOR_LENGTH,
            'INPUT_LENGTH': tokenizer.MAX_SEQUENCE_LENGTH,
            'TRAINABLE': False
        },

        'cnn': {
            'FILTERS': 16,
            'KERNEL': 3,
            'ACTIVATION': 'relu',
        },

        'max-pooling': {
            'POOL_SIZE': 2,
            'STRIDES': 1
        },

        'dropout': {
            'RATE': 0.3
        },

        'lstm': {
            'UNITS': 100,
        },

        'dense': {
            'UNITS': 3,
            'ACTIVATION': 'sigmoid'
        },

        'compile': {
            'LOSS': 'binary_crossentropy',
            'OPTIMIZER': 'adam'
        }
    }