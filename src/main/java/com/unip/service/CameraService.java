package com.unip.service;

import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.springframework.stereotype.Service;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import java.util.function.Consumer;

@Service
public class CameraService {

    private VideoCapture videoCapture;
    private volatile boolean cameraAtiva = false;
    private Thread cameraThread;

    public void startCamera(Consumer<Mat> frameCallback) {
        stopCamera();
        cameraAtiva = true;
        cameraThread = new Thread(() -> {
            // videoCapture = new VideoCapture(0, opencv_videoio.CAP_DSHOW);
            videoCapture = createVideoCapture();

            if (!videoCapture.isOpened()) {
                System.err.println("Não foi possível abrir a câmera!");
                return;
            }

            Mat frame = new Mat();
            while (cameraAtiva && videoCapture.read(frame)) {
                if (!frame.empty()) {
                    Mat frameRGB = new Mat();
                    cvtColor(frame, frameRGB, COLOR_BGR2RGB);
                    frameCallback.accept(frameRGB);
                }
                try {
                    Thread.sleep(33);
                } catch (InterruptedException e) {
                    break;
                }
        }
            videoCapture.release();
        });
        cameraThread.setDaemon(true);
        cameraThread.start();
    }

    private VideoCapture createVideoCapture() {
        String os = System.getProperty("os.name").toLowerCase();
        VideoCapture capture;

        if (os.contains("win")) {
            // Windows - DirectShow
            capture = new VideoCapture(0, opencv_videoio.CAP_DSHOW);
            System.out.println("Usando driver: Windows (DirectShow)");
        } else if (os.contains("mac")) {
            // macOS - AVFoundation
            capture = new VideoCapture(0, opencv_videoio.CAP_AVFOUNDATION);
            System.out.println("Usando driver: macOS (AVFoundation)");
        } else {
            // Linux/Unix - V4L2
            capture = new VideoCapture(0, opencv_videoio.CAP_V4L2);
            System.out.println("Usando driver: Linux (V4L2)");
        }

        // Padrão caso não funcione
        if (!capture.isOpened()) {
            System.out.println("Driver específico falhou, tentando auto-detect...");
            capture = new VideoCapture(0);
        }

        // Se ainda não abrir, tenta diferentes índices de câmera
        if (!capture.isOpened()) {
            System.out.println("Tentando câmeras alternativas...");
            for (int i = 1; i < 3; i++) {
                capture = new VideoCapture(i);
                if (capture.isOpened()) {
                    System.out.println("Câmera encontrada no índice: " + i);
                    break;
                }
            }
        }

        return capture;
    }

    public void stopCamera() {
        cameraAtiva = false;
        if (cameraThread != null)
            cameraThread.interrupt();
        if (videoCapture != null && videoCapture.isOpened())
            videoCapture.release();
    }

    public boolean isCameraActive() {
        return cameraAtiva;
    }

    public Mat captureFrame() {
        if (videoCapture != null && videoCapture.isOpened()) {
            Mat frame = new Mat();
            videoCapture.read(frame);
            return frame;
        }
        return null;
    }
}
