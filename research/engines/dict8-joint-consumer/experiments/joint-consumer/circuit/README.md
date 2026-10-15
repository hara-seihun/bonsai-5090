# Six-gate table-free 3-trit / 5-bit native-FP4 carrier

## Result and boundary

Start: three independent native nibbles in `{0,2,A}`. End: the consumer receives exactly those original nibbles, in the original order, without a runtime dictionary. Five bits are minimal (`16 < 27 <= 32`). This construction uses **six LOP3 gates** for five input bitplanes to six exact native magnitude/sign bitplanes.

This is a CPU-established Boolean circuit and an explicit physical lowering. There is no GPU execution, model/DLL execution, rental, download, serving mutation, or timing claim here. The engine writer owns integration. Their reported full-kernel sm_120a compile and forthcoming full fixture are separate evidence.

## Encoding and circuit

The code is `a + 2b + 4c + 8d + 16e`. A LOP3 immediate uses NVIDIA indexing `4*x + 2*y + z`.

```
M0 = LUT(a,d,e,0xd5)       // ~e | (a & d)
S0 = LUT(d,c,M0,0x82)     // M0 & ~(d ^ c)
M1 = LUT(b,d,e,0x97)
S1 = LUT(c,M1,0,0xc0)     // c & M1
M2 = LUT(b,M1,0,0x3c)     // b ^ M1
S2 = LUT(a,b,M1,0x24)
```

Each recovered nibble is `(M << 1) | (S << 3)`. All 32 raw codes decode to native-valid triplets; the image is all 27. Five aliases are deliberately reserved on the host: **3,7,22,23,28**. The canonical encoder chooses the least preimage. `encoding.csv` is the complete 27-entry source→code map; it is an inspection artifact, not a runtime lookup. The preparation encoder in `carrier.hpp` inverts the six gates in a fixed scan of 32 candidates, without a dictionary.

The checked source API returns one of `invalid_nibble`, `code_out_of_range`, `reserved_code`, or `CircuitInvariant`. There is no trap or silent repair. The unchecked device function requires host validation; the hardware's alias behavior is fully specified rather than undefined.

Six is optimal in the **one-result Boolean gate, separate output-bitplane grammar**, including DAG reuse: all six required output functions are distinct, nonconstant, and cannot be an input or its complement. Over the canonical 27 states, each sign has exactly nine ones and each magnitude eighteen. A raw five-bit coordinate (or its complement), after removing only five codewords, has between eleven and sixteen ones. Thus all six outputs require new gate results. The circuit attains six. This is not a global lower bound for arbitrary packed-word shifts/PRMT/integer ISA compositions.

## Eight sets/lane: literal 25-data-operation lowering

`Payload8` is serialized as four little-endian bytes `L`, followed by byte `H`: **5 bytes / 24 trits**. Do not memcpy the C++ struct (its ABI has padding).

For each set `j=0..7`, `L` nibble `j` contains low four code bits, and `H` bit `j` contains the fifth. Choose the three source trits as:

```
(source[j], source[8+j], source[16+j])
```

This is a producer-side grouping, not an arithmetic transform. The three output words contain original consecutive source nibbles `[0..7]`, `[8..15]`, `[16..23]` directly: **zero output PRMT/regroup operations**.

Align the important input bits at native magnitude positions `4*j+1`:

```
a=L<<1; b=L; c=L>>1; d=L>>2;
e=spread8(H)<<1;
```

`spread8` has three stages `(x | (x << k)) & mask`, each one shift plus one LOP3, for `(k,mask)=(12,000F000F),(6,03030303),(3,11111111)`.

Run the six gates on these unmasked words. Junk outside selected bit positions is harmless only because assembly explicitly masks:

```
mag = M & 0x22222222;
word = ((S << 2) & 0x88888888) | mag;
```

Assembly is one AND/LOP3, one shift, one fused LOP3 per word. Exact data invoice:

| Component | Shifts | LOP3 | Total |
|---|---:|---:|---:|
| L alignment | 3 | 0 | 3 |
| H spread + alignment | 4 | 3 | 7 |
| Boolean decoder | 0 | 6 | 6 |
| Three native words | 3 | 6 | 9 |
| **Total / 24 trits** | **10** | **15** | **25** |

`carrier_device.cuh` supplies literal immediate PTX LOP3 functions. Count excludes constant/register materialization, loads, addresses, predicates, output moves/stores, scales, cross-lane fragment ownership, synchronization and MMA. It is a source-level data invoice, not an observed SASS count. Both AND and XOR count as one LOP3-equivalent data operation. The functions contain no shared-memory or dictionary access.

Ordinary dense five-plane bytes require a different output bit transpose and do not inherit this 25-operation invoice. Storing each plane already spread in 32-bit holes would use 20 bytes/24 trits, destroying density. Regrouping cannot be silently omitted by choosing that format.

## Selected engine pair layout: 1728 bytes/warp tile

The engine writer chose pair-level native-word order `q=2*w+arm`. Group `q=0..29` into ten groups of three original native words. Each group uses one low-code word and one high-code byte. Grouping can cross arm labels in storage; decoder restores the original words before their unchanged consumer. No arithmetic across arms is introduced.

Two remaining native words (`q=30,31`) share one raw 32-bit carrier:

```
raw = gateWord | (upWord >> 1)       // preparation only
restoredGate = raw & 0xAAAAAAAA
restoredUp   = (raw << 1) & 0xAAAAAAAA
```

The original gate bits at nibble positions 1,3 and shifted-up bits at positions 0,2 are disjoint. The tail costs **three** recurring data operations, not two compact-tail decoders.

For 32 lanes:

- low-code words: `10 * 32 * 4 = 1280 B`;
- high-code bytes: `10 * 32 = 320 B`;
- raw tail words: `32 * 4 = 128 B`;
- total **1728 B**, versus **2048 B** for the original two-bit payload: **15.625% less**;
- decoder data work per lane/pair: `10*25 + 3 = 253` operations;
- output: **32 exact original native words**, 128 B/lane.

The 15.625% comparison is against the two-bit payload, not an uncompressed FP4-nibble stream (4096 B for these 32 lanes). Both use the same scales/metadata, which are outside these byte totals. Without grouping across arm labels, independent 128-trit operands can also use five groups plus a 16-bit compact tail; that alternative totals the same 54 B/pair but costs 270 decoder data operations. It is not the selected pair layout.

With SoA warp planes, a literal implementation reads ten U32 low words, ten U8 high bytes and one U32 raw word per lane: **21 load instructions / pair**, before any load reuse/combining by the engine. This is a load-instruction model, not a memory-transaction or bandwidth measurement. Whole-kernel shared staging, register lifetime and address costs must be priced in the engine.

For arbitrary independent blocks, padding must be explicit. Padding 128 trits to six full 24-trit groups uses 30 B, leaving only 6.25% savings versus 32 B. Ten groups plus the selected raw tail avoids that waste for the 256-trit pair. Pad source positions with exact native zero and discard only explicitly padded positions; never let padding change scale associations or fragment coordinates.

## Retained evidence and reproduction

```
cd ~/work/five-wide-native/joint-circuit-worker
g++ -O3 -std=c++17 -Wall -Wextra -Werror witness.cpp -o witness
./witness > encoding.csv 2> witness.log
~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean Carrier.lean
```

- `carrier.hpp`: host boundary, table-free scalar inversion, bitplane circuit, exact physical word decoder, raw-pair tail and alternative compact tail.
- `carrier_device.cuh`: native device functions, supplied for compile/integration, not GPU-executed here.
- `witness.cpp`, `witness.log`: all 32 raw codes; all 27 canonical roundtrips; five reserved errors; 216 physical single-slot roundtrips; all 256 spreading inputs; 28,672 two-slot patterns including aliases; compact-tail and selected raw-pair checks.
- `Carrier.lean`, `lean.log`: checked source coverage, canonical injectivity/count, encoder roundtrip, raw-native validity, five-bit capacity and output-function facts for the scoped six-gate lower bound. Standard Lean only, no `sorry`, no custom axiom and no downloaded dependency. These theorems cover the finite semantic circuit, not compiled CUDA/SASS or whole-engine execution.
- `search6dag.cpp`, `search6dag.log`: reproducible bounded CPU discovery and found six-gate DAG. Earlier one-level six-LUT solver attempts timed out; no negative theorem was inferred. A seven-gate circuit was found first and superseded by this six-gate DAG. No further search is needed for the integration candidate.

The observer-preserving claim is exact operand restoration. Scales, source coordinates, K ordering, MMA instruction grouping and accumulation schedule remain obligations of the consuming engine. Merely restoring the same multiset of values would not establish that contract. There is no performance claim.
