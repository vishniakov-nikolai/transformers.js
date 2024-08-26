const CACHE_DIR = 'C:\\Temp\\models\\';

main();

async function main() {
  const { pipeline, env } = await import('transformers.js');
  env.cacheDir = CACHE_DIR;
  const classificator = await pipeline(
    'image-classification',
    'Charles95/openvino_resnet50',
    {
      'cache_dir': CACHE_DIR,
      // 'local_files_only': true,
      'progress_callback': ({ status, name, file, progress, loaded, total }) => {
        if (!progress) return;

        process.stdout.clearLine();
        process.stdout.cursorTo(0);
        process.stdout.write(`== progress of '${file}': ${Math.ceil(progress)}%`);
      },
      'model_file_name': ['openvino_model.xml', 'openvino_model.bin'],
      device: 'AUTO',
      inferenceCallback,
    },
  );

  let output;

  for (const i of new Array(5).fill(null)) {
    output = await classificator('./assets/coco.jpg', {
      threshold: 0.5,
      percentage: true,
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
