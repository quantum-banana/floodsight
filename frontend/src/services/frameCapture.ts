export interface CapturedJpeg {
  blob: Blob;
  width: number;
  height: number;
}

export function scaledDimensions(
  width: number,
  height: number,
  maxWidth: number,
  maxHeight: number,
) {
  const scale = Math.min(1, maxWidth / width, maxHeight / height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export function captureJpeg(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  options: { maxWidth: number; maxHeight: number; quality: number },
): Promise<CapturedJpeg> {
  if (!video.videoWidth || !video.videoHeight) {
    return Promise.reject(new Error("The media source has no decoded video dimensions yet."));
  }
  const dimensions = scaledDimensions(
    video.videoWidth,
    video.videoHeight,
    options.maxWidth,
    options.maxHeight,
  );
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) return Promise.reject(new Error("Canvas frame capture is unavailable."));
  context.drawImage(video, 0, 0, dimensions.width, dimensions.height);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("The browser could not encode the captured frame."));
          return;
        }
        resolve({ blob, ...dimensions });
      },
      "image/jpeg",
      options.quality,
    );
  });
}
