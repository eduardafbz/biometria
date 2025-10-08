package com.unip.service;

import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.opencv_core.*;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Optional;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class FaceDetectionService {
    private CascadeClassifier faceDetector;
    
    public FaceDetectionService() {
        try {
            InputStream is = getClass().getResourceAsStream("/haarcascade_frontalface_default.xml");
            if (is == null) {
                System.err.println("Classificador Haar não encontrado! Baixar arquivo:");
                System.err.println("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml");
                return;
            }
            
            File tempFile = File.createTempFile("haarcascade", ".xml");
            Files.copy(is, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            tempFile.deleteOnExit();
            
            faceDetector = new CascadeClassifier(tempFile.getAbsolutePath());
            if (faceDetector == null || faceDetector.isNull()) {
                System.err.println("Erro ao carregar classificador facial!");
            } else {
                System.out.println("Detector facial carregado com sucesso!");
            }
        } catch (Exception e) {
            System.err.println("Erro ao inicializar detector facial: " + e.getMessage());
        }
    }
    
    public Optional<Mat> detectAndExtractFace(Mat image) {
        if (faceDetector == null || faceDetector.isNull()) {
            System.err.println("Detector facial não inicializado!");
            return Optional.empty();
        }
        
        try {
            Mat gray = new Mat();
            cvtColor(image, gray, COLOR_BGR2GRAY);
            
            // Equaliza histograma para melhor detecção
            equalizeHist(gray, gray);
            
            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(gray, faces, 1.1, 3, 0, new Size(30, 30), new Size());
            
            System.out.println("Faces detectadas: " + faces.size());
            
            if (faces.size() > 0) {
                Rect faceRect = faces.get(0);
                Mat face = new Mat(gray, faceRect);
                
                // Redimensiona para tamanho padrão
                Mat resizedFace = new Mat();
                resize(face, resizedFace, new Size(150, 150));
                
                return Optional.of(resizedFace);
            }
        } catch (Exception e) {
            System.err.println("Erro na detecção facial: " + e.getMessage());
        }
        
        return Optional.empty();
    }
}