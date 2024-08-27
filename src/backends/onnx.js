/**
 * @file Handler file for choosing the correct version of ONNX Runtime, based on the environment.
 * Ideally, we could import the `onnxruntime-web` and `onnxruntime-node` packages only when needed,
 * but dynamic imports don't seem to work with the current webpack version and/or configuration.
 * This is possibly due to the experimental nature of top-level await statements.
 * So, we just import both packages, and use the appropriate one based on the environment:
 *   - When running in node, we use `onnxruntime-node`.
 *   - When running in the browser, we use `onnxruntime-web` (`onnxruntime-node` is not bundled).
 *
 * This module is not directly exported, but can be accessed through the environment variables:
 * ```javascript
 * import { env } from '@xenova/transformers';
 * console.log(env.backends.onnx);
 * ```
 *
 * @module backends/onnx
 */

const {
    getModelFile,
} = require('../utils/hub.js');

// NOTE: Import order matters here. We need to import `onnxruntime-node` before `onnxruntime-web`.
// In either case, we select the default export if it exists, otherwise we use the named export.
const ONNX_NODE = require('onnxruntime-node');

/** @type {import('onnxruntime-web')} The ONNX runtime module. */
let ONNX;

const executionProviders = [
    // 'webgpu',
    'wasm'
];

if (typeof process !== 'undefined' && process?.release?.name === 'node') {
    // Running in a node-like environment.
    // ONNX = ONNX_NODE.default ?? ONNX_NODE;
    ONNX = ONNX_NODE;

    // Add `cpu` execution provider, with higher precedence that `wasm`.
    executionProviders.unshift('cpu');

}
// else {
//     // Running in a browser-environment
//     ONNX = ONNX_WEB.default ?? ONNX_WEB;

//     // SIMD for WebAssembly does not operate correctly in some recent versions of iOS (16.4.x).
//     // As a temporary fix, we disable it for now.
//     // For more information, see: https://github.com/microsoft/onnxruntime/issues/15644
//     const isIOS = typeof navigator !== 'undefined' && /iP(hone|od|ad).+16_4.+AppleWebKit/.test(navigator.userAgent);
//     if (isIOS) {
//         ONNX.env.wasm.simd = false;
//     }
// }

async function create(pretrained_model_name_or_path, fileName, options) {
    let modelFileName = `onnx/${fileName}${options.quantized ? '_quantized' : ''}.onnx`;
    let buffer = await getModelFile(pretrained_model_name_or_path, modelFileName, true, options);

    try {
        return await ONNX.InferenceSession.create(buffer, {
            executionProviders,
        });
    } catch (err) {
        // If the execution provided was only wasm, throw the error
        if (executionProviders.length === 1 && executionProviders[0] === 'wasm') {
            throw err;
        }

        console.warn(err);
        console.warn(
            'Something went wrong during model construction (most likely a missing operation). ' +
            'Using `wasm` as a fallback. '
        )
        return await ONNX.InferenceSession.create(buffer, {
            executionProviders: ['wasm']
        });
    }
}

module.exports = {
    create,
    executionProviders,
    ONNX,
};
