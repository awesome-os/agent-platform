import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { describe, it } from 'node:test';
import {
  B8G_ABI_MAGIC_V3,
  B8SW_HEADER_BYTES,
  B8SW_MAGIC,
  B8SW_VERSION,
  parseB8pHeader,
  SpinWaveMoEModel,
  type TrainingSample,
} from './spinwave-moe.mts';

const CLASS_LABELS = ['compiler-build', 'cas-storage', 'display-ipc', 'pkg-manager'] as const;

const TRAIN_DATASET: readonly TrainingSample[] = [
  // Class 0: compiler-build
  { input: 'holo-compiler compile zig cc target x86_64-linux-musl -O3', targetClass: 0 },
  { input: 'b8g-cxx unity batch clang++ pch incremental build ninja', targetClass: 0 },
  { input: 'compile sectium-jvm wasm32-wasi aot binary linker lld', targetClass: 0 },
  { input: 'cross compile native c11 shared library without wsl', targetClass: 0 },
  { input: 'build sectium-es elf64 win64 pe zero-copy runtime', targetClass: 0 },
  { input: 'run b8g-ninja-driver sub-tu code-block compilation cache', targetClass: 0 },
  { input: 'compile c++ object files with shared transitive pch', targetClass: 0 },
  { input: 'holo-compiler tier-a native build c-abi interface stub', targetClass: 0 },
  { input: 'invoke zig cc to link static musl executable -O2', targetClass: 0 },
  { input: 'gn gen out/release and ninja compile v8 snapshot', targetClass: 0 },
  { input: 'compile erasable typescript bundle and native wasm module', targetClass: 0 },
  { input: 'link object archive libsectium.a with lld-link flags', targetClass: 0 },
  { input: 'rebuild modified translation unit with holo-compiler', targetClass: 0 },
  { input: 'compile kernel initramfs c11 binary 5kb target', targetClass: 0 },
  { input: 'clang c++20 optimization pass -flto native codegen', targetClass: 0 },
  { input: 'build native addon with zig c++ cross toolchain', targetClass: 0 },

  // Class 1: cas-storage
  { input: 'store merkle tree blob in .b8p content addressable cas pack', targetClass: 1 },
  { input: 'uvs-merkle snapshot commit hash deduplication cas store', targetClass: 1 },
  { input: 'checkpoint warm-ram b8g_arena_v1 snapshot into .b8p archive', targetClass: 1 },
  { input: 'verify sha256 content hash in .b8g-cas object pack', targetClass: 1 },
  { input: 'gc prune unreferenced blobs from cas content-addressed storage', targetClass: 1 },
  { input: 'read immutable trie node from .b8p wasm cas pack', targetClass: 1 },
  { input: 'hydrate merkle root tree from local cas object store', targetClass: 1 },
  { input: 'compress delta blocks into content-addressed .b8p volume', targetClass: 1 },
  { input: 'index cas chunks by blake3 digest for zero-copy mount', targetClass: 1 },
  { input: 'rolling release manifest o(1) rollback to prior .b8p digest', targetClass: 1 },
  { input: 'sync cas bucket objects across remote merkle mirrors', targetClass: 1 },
  { input: 'inspect .b8p archive table of contents and chunk hashes', targetClass: 1 },
  { input: 'deduplicate identical artifact chunks inside .b8g-cas store', targetClass: 1 },
  { input: 'verify merkle proof integrity for stored cas blob', targetClass: 1 },
  { input: 'compact fragmented cas packs into single immutable .b8p', targetClass: 1 },
  { input: 'restore warm checkpoint arena from cas blob digest', targetClass: 1 },

  // Class 2: display-ipc
  { input: 'connect named pipe \\\\.\\pipe\\sectium-dev ndjson display sink', targetClass: 2 },
  { input: 'route sectium devtools plane over channel command ttys0', targetClass: 2 },
  { input: 'stream pure-art frame to sectium canvas appliance socket', targetClass: 2 },
  { input: 'open /tmp/sectium-dev.sock unix domain socket ipc channel', targetClass: 2 },
  { input: 'attach kitty-ink-sink tui renderer to host device boundary', targetClass: 2 },
  { input: 'dispatch cdp pipe message over \\\\.\\pipe\\b8g-host-bus', targetClass: 2 },
  { input: 'hot-swap display sink channel over tcp mtls transport', targetClass: 2 },
  { input: 'zero-copy sharedarraybuffer b8g_arena_v1 frame ipc ring', targetClass: 2 },
  { input: 'send power-button wake frame to sectium display device', targetClass: 2 },
  { input: 'subscribe createEventStream cross-tab devtools ipc bus', targetClass: 2 },
  { input: 'write ndjson frame payload to host-bus.sock display pipe', targetClass: 2 },
  { input: 'configure uefi lo-res framebuffer over sectium device pipe', targetClass: 2 },
  { input: 'multiplex command ttys0 and display channels on host pipe', targetClass: 2 },
  { input: 'render html5 b8g custom element over sectium canvas ipc', targetClass: 2 },
  { input: 'flush vsync surface buffer to sectium display appliance', targetClass: 2 },
  { input: 'inspect host pipe ipc latency on \\\\.\\pipe\\b8g-host-bus', targetClass: 2 },

  // Class 3: pkg-manager
  { input: 'pnpm install workspace dependencies with pnx runner', targetClass: 3 },
  { input: 'pack node_modules into deps.b8p via @awesome-os/b8g-pm', targetClass: 3 },
  { input: 'rewrite npx invocation to pnx through pnpm-shim virtualization', targetClass: 3 },
  { input: 'resolve stealifylang direct source import install-if-missing', targetClass: 3 },
  { input: 'update workspace package.json dependencies and pnpm-lock.yaml', targetClass: 3 },
  { input: 'run pnx biome check across monorepo packages', targetClass: 3 },
  { input: 'audit installed package versions and prune unused deps', targetClass: 3 },
  { input: 'link local workspace package @awesome-os/bin-shim via pnpm', targetClass: 3 },
  { input: 'install missing github url module with stealify lookup', targetClass: 3 },
  { input: 'remove legacy package-lock.json and yarn.lock bloat', targetClass: 3 },
  { input: 'pnpm filter @awesome-os/b8g-pm run test workspace script', targetClass: 3 },
  { input: 'verify zero npx usage across package scripts and shims', targetClass: 3 },
  { input: 'resolve transitive npm dependency graph into deps.b8p pack', targetClass: 3 },
  { input: 'corepack pnpm version pin for workspace package management', targetClass: 3 },
  { input: 'list outdated workspace packages with pnpm outdated', targetClass: 3 },
  { input: 'add devdependency to package.json using pnpm add -D', targetClass: 3 },
];

// Held-out validation set (unseen during training)
const VALIDATION_DATASET: readonly TrainingSample[] = [
  // Class 0: compiler-build
  { input: 'holo-compiler native cross-compile zig cc -O3 linux binary', targetClass: 0 },
  { input: 'incremental c++ pch compilation via b8g-cxx ninja driver', targetClass: 0 },
  { input: 'build wasm32-wasi aot module and link with lld', targetClass: 0 },
  { input: 'compile c11 runtime target win64 pe without wsl', targetClass: 0 },
  { input: 'clang++ unity batch compile native object files', targetClass: 0 },
  { input: 'build static musl executable with zig cc toolchain', targetClass: 0 },

  // Class 1: cas-storage
  { input: 'store immutable merkle blob inside .b8p cas pack', targetClass: 1 },
  { input: 'verify sha256 chunk hash in .b8g-cas object store', targetClass: 1 },
  { input: 'checkpoint warm-ram arena snapshot to cas .b8p archive', targetClass: 1 },
  { input: 'rollback manifest to previous .b8p digest in cas storage', targetClass: 1 },
  { input: 'deduplicate merkle tree chunks and prune unreferenced blobs', targetClass: 1 },
  { input: 'hydrate content-addressed trie node from .b8p volume', targetClass: 1 },

  // Class 2: display-ipc
  { input: 'stream ndjson frame over \\\\.\\pipe\\sectium-dev display sink', targetClass: 2 },
  { input: 'route devtools event stream on channel command ttys0 pipe', targetClass: 2 },
  { input: 'connect kitty-ink-sink tui over /tmp/sectium-dev.sock ipc', targetClass: 2 },
  { input: 'send cdp message across \\\\.\\pipe\\b8g-host-bus socket', targetClass: 2 },
  { input: 'hot-swap sectium canvas display appliance over tcp mtls', targetClass: 2 },
  { input: 'render frame buffer via host device boundary ipc channel', targetClass: 2 },

  // Class 3: pkg-manager
  { input: 'pnpm install monorepo packages and run pnx script', targetClass: 3 },
  { input: 'pack workspace node_modules into deps.b8p with b8g-pm', targetClass: 3 },
  { input: 'replace npx command with pnx via pnpm-shim', targetClass: 3 },
  { input: 'update package.json and pnpm-lock.yaml dependencies', targetClass: 3 },
  { input: 'resolve missing stealify module import from workspace packages', targetClass: 3 },
  { input: 'prune unused npm dependencies using pnpm filter', targetClass: 3 },
];

describe('SpinWaveMoEModel', () => {
  it('maintains unit-circle phasor & expert-key invariants and Top-K sparse activation', () => {
    const model = new SpinWaveMoEModel({
      vocabSize: 256,
      dPhasor: 16,
      numExperts: 16,
      topK: 2,
      classLabels: CLASS_LABELS,
      seed: 0x42b857,
    });

    // Verify all expert phase keys lie on the unit circle
    for (let i = 0; i < model.numExperts * model.dPhasor; i++) {
      const magSq = model.expRe[i]! ** 2 + model.expIm[i]! ** 2;
      assert.ok(Math.abs(magSq - 1) < 1e-5, `Expert key ${i} not on unit circle: |z|^2=${magSq}`);
    }

    const pred = model.predict('holo-compiler zig cc build wasm32-wasi');
    assert.equal(pred.activeExperts.length, 2, 'Top-K router must activate exactly k=2 experts');
    assert.equal(pred.gateWeights.length, 2);
    assert.notEqual(pred.activeExperts[0], pred.activeExperts[1], 'Active experts must be distinct');

    // Verify gate weights sum to 1
    const gateSum = pred.gateWeights[0]! + pred.gateWeights[1]!;
    assert.ok(Math.abs(gateSum - 1) < 1e-5, `Gate weights should sum to 1, got ${gateSum}`);

    // Verify encoded query phasor lies on the unit circle
    for (let d = 0; d < model.dPhasor; d++) {
      const magSq = pred.phasorRe[d]! ** 2 + pred.phasorIm[d]! ** 2;
      assert.ok(Math.abs(magSq - 1) < 1e-5, `Phasor dim ${d} not on unit circle: |q|^2=${magSq}`);
    }

    // Verify activeExperts are indeed the top-2 coherences
    const sortedExperts = Array.from(pred.expertCoherences)
      .map((coh, idx) => ({ coh, idx }))
      .sort((a, b) => b.coh - a.coh);
    assert.equal(pred.activeExperts[0], sortedExperts[0]!.idx);
    assert.equal(pred.activeExperts[1], sortedExperts[1]!.idx);
  });

  it('trains deterministically on 4-class Awesome-OS dataset with >= 0.90 held-out accuracy, quantizes to < 64 KB .b8p with >= 0.88 accuracy, and runs < 0.5 ms/sample', () => {
    const model = new SpinWaveMoEModel({
      vocabSize: 256,
      dPhasor: 16,
      numExperts: 16,
      topK: 2,
      classLabels: CLASS_LABELS,
      seed: 0x20261124,
    });

    const trainMetrics = model.train(TRAIN_DATASET, {
      epochs: 45,
      learningRate: 0.2,
      lrDecay: 0.97,
      weightDecay: 1e-4,
      seed: 0x99887766,
    });

    assert.ok(
      trainMetrics.finalLoss < trainMetrics.lossHistory[0]!,
      `Training loss must decrease: ${trainMetrics.lossHistory[0]} -> ${trainMetrics.finalLoss}`,
    );

    // Held-out validation accuracy (FP32)
    const fp32Eval = model.evaluate(VALIDATION_DATASET);
    assert.ok(
      fp32Eval.accuracy >= 0.9,
      `Expected FP32 held-out accuracy >= 0.90, got ${(fp32Eval.accuracy * 100).toFixed(2)}% (${fp32Eval.correct}/${fp32Eval.total})`,
    );

    // Determinism check: identical seed produces bit-identical weights
    const modelClone = new SpinWaveMoEModel({
      vocabSize: 256,
      dPhasor: 16,
      numExperts: 16,
      topK: 2,
      classLabels: CLASS_LABELS,
      seed: 0x20261124,
    });
    modelClone.train(TRAIN_DATASET, {
      epochs: 45,
      learningRate: 0.2,
      lrDecay: 0.97,
      weightDecay: 1e-4,
      seed: 0x99887766,
    });
    assert.deepEqual(model.encRe, modelClone.encRe);
    assert.deepEqual(model.expRe, modelClone.expRe);

    // Unit-circle invariant preserved after training
    for (let i = 0; i < model.numExperts * model.dPhasor; i++) {
      const magSq = model.expRe[i]! ** 2 + model.expIm[i]! ** 2;
      assert.ok(Math.abs(magSq - 1) < 1e-4, `Trained expert key ${i} drifted off unit circle`);
    }

    // INT8 .b8p serialization & header validation
    const b8p = model.toB8pBuffer();
    assert.ok(
      b8p.byteLength < 64 * 1024,
      `Expected .b8p size < 65536 bytes, got ${b8p.byteLength} bytes`,
    );

    const header = parseB8pHeader(b8p);
    assert.equal(header.magic, B8SW_MAGIC);
    assert.equal(header.version, B8SW_VERSION);
    assert.equal(header.abiMagic, B8G_ABI_MAGIC_V3);
    assert.equal(header.vocabSize, 256);
    assert.equal(header.dPhasor, 16);
    assert.equal(header.numExperts, 16);
    assert.equal(header.topK, 2);
    assert.equal(header.numClasses, 4);
    assert.ok(header.totalBytes >= B8SW_HEADER_BYTES + 4);

    // Reload from .b8p and verify quantized accuracy >= 0.88
    const reloaded = SpinWaveMoEModel.fromB8pBuffer(b8p);
    assert.deepEqual(reloaded.classLabels, CLASS_LABELS);
    const int8Eval = reloaded.evaluate(VALIDATION_DATASET);
    assert.ok(
      int8Eval.accuracy >= 0.88,
      `Expected INT8 quantized held-out accuracy >= 0.88, got ${(int8Eval.accuracy * 100).toFixed(2)}% (${int8Eval.correct}/${int8Eval.total})`,
    );

    // Verify Top-K sparsity on reloaded model
    const samplePred = reloaded.predict(VALIDATION_DATASET[0]!.input);
    assert.equal(samplePred.activeExperts.length, 2);

    // Measure inference latency (< 0.5 ms/sample = 500 µs/sample)
    const warmupIters = 100;
    for (let i = 0; i < warmupIters; i++) {
      reloaded.predict(VALIDATION_DATASET[i % VALIDATION_DATASET.length]!.input);
    }

    const benchIters = 2000;
    const t0 = performance.now();
    for (let i = 0; i < benchIters; i++) {
      reloaded.predict(VALIDATION_DATASET[i % VALIDATION_DATASET.length]!.input);
    }
    const elapsedMs = performance.now() - t0;
    const msPerSample = elapsedMs / benchIters;
    const usPerSample = msPerSample * 1000;

    assert.ok(
      msPerSample < 0.5,
      `Expected inference latency < 0.5 ms/sample, got ${msPerSample.toFixed(4)} ms/sample (${usPerSample.toFixed(2)} µs/sample)`,
    );

    // Log metrics for test visibility
    console.log(
      `[SpinWaveMoE] FP32 val=${(fp32Eval.accuracy * 100).toFixed(1)}% (${fp32Eval.correct}/${fp32Eval.total}) | ` +
        `INT8 val=${(int8Eval.accuracy * 100).toFixed(1)}% (${int8Eval.correct}/${int8Eval.total}) | ` +
        `.b8p=${b8p.byteLength} B | latency=${usPerSample.toFixed(2)} us/sample`,
    );
  });

  it('rejects corrupted .b8p checksums and invalid headers', () => {
    const model = new SpinWaveMoEModel({ classLabels: CLASS_LABELS });
    const b8p = model.toB8pBuffer();

    const corrupted = new Uint8Array(b8p);
    corrupted[B8SW_HEADER_BYTES + 10] = (corrupted[B8SW_HEADER_BYTES + 10]! ^ 0xff) & 0xff;
    assert.throws(() => SpinWaveMoEModel.fromB8pBuffer(corrupted), /checksum mismatch/i);

    const badMagic = new Uint8Array(b8p);
    badMagic[0] = 0x58; // 'X'
    assert.throws(() => SpinWaveMoEModel.fromB8pBuffer(badMagic), /Invalid B8SW magic/i);
  });

  it('crystallize() immediately corrects a novel unseen sample', () => {
    const model = new SpinWaveMoEModel({
      vocabSize: 256,
      dPhasor: 16,
      numExperts: 16,
      topK: 2,
      classLabels: CLASS_LABELS,
      seed: 0x20261124,
    });
    model.train(TRAIN_DATASET, { epochs: 35, learningRate: 0.2, seed: 0x123456 });

    // Novel domain slang unseen in training
    const novelSample = 'zyxwvut qrstuvwx magnonic-quantum-flux-capacitor-9000';
    const before = model.predict(novelSample);
    // Pick a target class different from whatever the model currently predicts
    const targetClass = (before.classIndex + 1) % CLASS_LABELS.length;
    assert.notEqual(before.classIndex, targetClass);

    const result = model.crystallize(novelSample, targetClass, 12, 0.25);
    assert.equal(result.beforeClass, before.classIndex);
    assert.equal(result.afterClass, targetClass);
    assert.ok(
      result.afterConfidence > 0.7,
      `Expected high confidence after crystallization, got ${result.afterConfidence.toFixed(3)}`,
    );

    const after = model.predict(novelSample);
    assert.equal(after.classIndex, targetClass);
    assert.equal(after.classLabel, CLASS_LABELS[targetClass]);
  });

  it('supports up to 64 experts and validates config bounds', () => {
    const largeMoE = new SpinWaveMoEModel({
      vocabSize: 256,
      dPhasor: 16,
      numExperts: 64,
      topK: 4,
      classLabels: CLASS_LABELS,
    });
    const pred = largeMoE.predict('compile zig cc binary');
    assert.equal(pred.activeExperts.length, 4);

    const b8p = largeMoE.toB8pBuffer();
    assert.ok(b8p.byteLength < 64 * 1024, `64-expert .b8p should be < 64 KB, got ${b8p.byteLength}`);
    const reloaded = SpinWaveMoEModel.fromB8pBuffer(b8p);
    assert.equal(reloaded.numExperts, 64);
    assert.equal(reloaded.topK, 4);

    assert.throws(() => new SpinWaveMoEModel({ numExperts: 65 }), /numExperts/);
    assert.throws(() => new SpinWaveMoEModel({ numExperts: 8, topK: 9 }), /topK/);
  });
});
