const { pipeline, env } = require('transformers.js');

const CACHE_DIR = 'C:\\Temp\\models1\\';

main();

const modelsByTask = {
  // topk as limit number of results
  'image-classification': [
    'Charles95/openvino_resnet50',
    'IlyasMoutawwakil/vit-with-hidden_states',
    'helenai/microsoft-swin-tiny-patch4-window7-224-ov',
    {
      name: 'Xenova/convnextv2-tiny-1k-224',
      files: ['onnx/model.onnx'],
    },
  ],
  'image-segmentation': [
    {
       name: 'Xenova/segformer_b0_clothes',
       files: ['onnx/model.onnx'],
    },
    {
       name: 'Xenova/segformer-b0-finetuned-cityscapes-640-1280',
       files: ['onnx/model.onnx'],
    },
  ],
  'object-detection': [
    {
      name: 'Xenova/yolos-tiny',
      files: ['onnx/model.onnx']
    },
  ],
  'image-to-text': [
    {
      name: 'Xenova/vit-gpt2-image-captioning',
      files: ['onnx/model.onnx'],
    },
  ],
  'image-to-image': [],
  'depth-estimation': [],
  'text-generation': [
    'helenai/gpt2-ov',
  ],
};

async function main() {
  // const { pipeline, env } = await import('transformers.js');
  env.cacheDir = CACHE_DIR;
  // env.useFSCache = true;
  const generator = await pipeline(
    'image-classification',
    'Charles95/openvino_resnet50',
    {
      // 'progress_callback': ({ status, name, file, progress, loaded, total }) => {
      //   if (!progress) return;

      //   process.stdout.clearLine();
      //   process.stdout.cursorTo(0);
      //   process.stdout.write(`== progress of '${file}': ${Math.ceil(progress)}%`);
      // },
      'model_file_name': ['openvino_model.xml', 'openvino_model.bin'],
      // 'model_file_name': ['openvino_encoder_model.xml', 'openvino_encoder_model.bin'],
      // 'model_file_name': ['onnx/model_quantized.onnx'],
      // 'model_file_name': ['onnx/model.onnx'],
      // 'model_file_name': [],
      device: 'AUTO',
      inferenceCallback,
    },
  );

  let output;
  const times = 1;

  for (const _ of new Array(times).fill(null)) {
    output = await generator('./assets/pat.png', {
      // src_lang: 'ru',
      // tgt_lang: 'en',
      topk: 5,
      // temperature: 4,
      // max_new_tokens: 200,
      // 'callback_function': x => {
      //   console.log({
      //     output: generator.tokenizer.decode(x[0]['output_token_ids'])
      //   });
      // }
    });
  }

  console.log(output)
}

function inferenceCallback(nanosec) {
  console.log(`=== Inference time: ${formatNanoseconds(nanosec)}ms`)
}

function formatNanoseconds(bigNumber) {
  return Math.floor(Number(bigNumber) / 1000000);
}
