LONG_SCHEDULE = {
    'step_values': [400000, 600000, 800000, 1000000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 1200000,
}

SUBPIXEL_SCHEDULE = {
    'step_values': [8000, 11000, 14000, 17000],
    #'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625], # (1) jy
    'learning_rates': [0.00005, 0.000025, 0.0000125, 0.00000625, 0.000003125], # (2) jwlim
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 20000,
}

FINETUNE_SCHEDULE = {
    # TODO: Finetune schedule
}
