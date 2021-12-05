from eolearn.io import S2L1CWCSInput, SentinelHubInputTask, S2L2AWCSInput

# S2L1CWCSInput was replaced in eo-learn v0.9.0, need to update for new classes

input_task = S2L2AWCSInput('BANDS-S2-L1C', resx='10m', resy='10m', maxcc=0.8)
add_l2a = S2L2AWCSInput(layer='BANDS-S2-L2A')
true_color  = S2L1CWCSInput('TRUE-COLOR-S2-L1C')
SENT_SAT_CLASSICATIONS = S2L2AWCSInput("SCENE_CLASSIFICATION")