def setup_params(matrix, tokenizer):
    return {
        'embedding': {
            'OUTPUT_DIM': matrix.EMBEDDING_VECTOR_LENGTH,
            'INPUT_LENGTH': tokenizer.MAX_SEQUENCE_LENGTH,
            'TRAINABLE': False
        },

        'cnn': {
            'FILTERS': 128,
            'KERNEL': 3,
            'ACTIVATION': 'relu',
        },

        'max-pooling': {
            'POOL_SIZE': 2
        },

        'lstm': {
            'UNITS': 100,
            'DROPOUT': 0.8
        },

        'dense': {
            'UNITS': 3,
            'ACTIVATION': 'softmax'
        },

        'compile': {
            'LOSS': 'categorical_crossentropy',
            'OPTIMIZER': 'adam',
            'METRICS': ['accuracy']
        }
    }