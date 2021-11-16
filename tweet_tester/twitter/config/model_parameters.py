def setup_params(matrix, tokenizer):
    return {
        'embedding': {
            'OUTPUT_DIM': matrix.EMBEDDING_VECTOR_LENGTH,
            'INPUT_LENGTH': tokenizer.MAX_SEQUENCE_LENGTH,
            'TRAINABLE': False
        },

        'cnn': {
            'FILTERS': 64,
            'KERNEL': 5,
            'ACTIVATION': 'relu',
        },

        'max-pooling': {
            'POOL_SIZE': 3,
            'STRIDES': 1
        },

        'dropout': {
            'RATE': 0.2
        },

        'lstm': {
            'UNITS': 100,
        },

        'dense': {
            'UNITS': 3,
            'ACTIVATION': 'softmax'
        },

        'compile': {
            'LOSS': 'categorical_crossentropy',
            'OPTIMIZER': 'adam'
        }
    }