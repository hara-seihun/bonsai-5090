from h1_graph_contract import warmup_decode, graph_observation

class Lifecycle:
    def __init__(self, enabled=True):
        self.calls = []; self.hits = 0; self.captures = 0; self.enabled = enabled
    def step(self, slots, inputs, positions):
        self.calls.append((slots.copy(), inputs.copy(), positions.copy()))
        if self.enabled:
            if len(self.calls) == 3: self.captures += 1
            elif len(self.calls) > 3: self.hits += 1
        return [x + 1 for x in inputs]

slots = list(range(64)); inputs = [10] * 64; positions = [100] * 64
original = Lifecycle()
for _ in range(2): original.step(slots, inputs, positions)
assert not graph_observation(original.hits, original.captures, 0, 0, 2)['graph_steps_accepted']
reference = Lifecycle()
a, p, wi, wt = warmup_decode(reference.step, slots, inputs, positions, None)
assert len(reference.calls) == 3 and reference.captures == 1 and reference.hits == 0
assert a == [13] * 64 and p == [103] * 64
bh, bc = reference.hits, reference.captures
for _ in range(2): reference.step(slots, a, p)
assert graph_observation(reference.hits, reference.captures, bh, bc, 2)['graph_steps_accepted']
repeat = Lifecycle()
b, q, wri, wrt = warmup_decode(repeat.step, slots, inputs, positions, wt)
assert (b, q, wri, wrt) == (a, p, wi, wt)
changed = Lifecycle()
forced = [[30] * 64, [40] * 64, [50] * 64]
b, q, wci, wct = warmup_decode(changed.step, slots, inputs, positions, forced)
assert [row[1][0] for row in changed.calls] == [10, 30, 40] and b == [50] * 64
assert wct != forced
assert not graph_observation(0, 0, 0, 0, 2)['graph_steps_accepted']
assert not graph_observation(1, 1, 0, 0, 2)['graph_steps_accepted']
assert not graph_observation(5, 7, 5, 7, 2)['graph_steps_accepted']
assert graph_observation(7, 7, 5, 7, 2)['graph_steps_accepted']
assert inputs == [10] * 64 and positions == [100] * 64
print('PASS: v1 two-visit failure, three-visit warmup, exact cached-observation counts, teacher forcing, untouched inputs, absent/mixed/preexisting-counter refusals')
