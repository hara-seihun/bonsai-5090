def warmup_decode(step, slots, inputs, positions, teacher):
    warmup_inputs = []
    warmup_tokens = []
    for i in range(3):
        warmup_inputs.append(inputs.copy())
        output = step(slots, inputs, positions)
        warmup_tokens.append(output)
        inputs = output if teacher is None else teacher[i]
        positions = [p + 1 for p in positions]
    return inputs, positions, warmup_inputs, warmup_tokens


def graph_observation(hits, captures, before_hits, before_captures, steps):
    observed = {'graph_hits_decode': hits - before_hits,
                'graph_captures_decode': captures - before_captures}
    observed['graph_steps_accepted'] = (steps > 0 and
        observed['graph_hits_decode'] == steps and
        observed['graph_captures_decode'] == 0)
    return observed
