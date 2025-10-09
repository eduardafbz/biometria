package com.unip.service;

import com.unip.model.Role;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.springframework.stereotype.Service;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.DoublePointer;

import java.io.*;
import java.util.*;
import java.util.function.Consumer;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;

@Service
public class FaceService {
    private final FaceRecognizer faceRecognizer;
    private final Map<Integer, String> idToNameMap = new HashMap<>();
    private final Map<Integer, String> idToEmailMap = new HashMap<>();
    private final Map<Integer, Role> idToRoleMap = new HashMap<>();
    private int nextId = 0;

    private final String FACES_DIR = "faces";
    private final String LABELS_FILE = FACES_DIR + "/labels.txt";

    private static final double LIMIAR_DUPLICATA = 50.0;
    private static final double LIMIAR_RECONHECIMENTO = 60.0;

    public FaceService() {
        faceRecognizer = LBPHFaceRecognizer.create();
        loadLabels();
        retrainModel();
    }

    public void register(Mat face, String personName, String email, Role role, Consumer<String> callback) {
        Mat grayFace = new Mat();
        cvtColor(face, grayFace, COLOR_BGR2GRAY);

        // Liminar mais alta - Evita falsos positivos
        double LIMIAR_DUPLICATA = 50.0;

        if (!idToNameMap.isEmpty()) {
            try {
                IntPointer label = new IntPointer(1);
                DoublePointer confidence = new DoublePointer(1);
                faceRecognizer.predict(grayFace, label, confidence);

                double conf = confidence.get(0);
                System.out.println("DEBUG REGISTER - Confiança: " + conf + ", Limiar: " + LIMIAR_DUPLICATA);

                if (conf < LIMIAR_DUPLICATA) {
                    String existingName = idToNameMap.get(label.get(0));
                    callback.accept("❌ Erro: Rosto já registrado como '" + existingName + "'");
                    return;
                }
            } catch (Exception e) {
                System.out.println("Modelo não treinado, primeiro registro...");
            }
        }

        // Verificar se email já existe
        if (idToEmailMap.containsValue(email)) {
            callback.accept("❌ Erro: Email já registrado");
            return;
        }

        int personId = nextId++;
        idToNameMap.put(personId, personName);
        idToEmailMap.put(personId, email);
        idToRoleMap.put(personId, role);

        File personDir = new File(FACES_DIR, String.valueOf(personId));
        personDir.mkdirs();
        String filename = new File(personDir, System.currentTimeMillis() + ".png").getAbsolutePath();
        opencv_imgcodecs.imwrite(filename, grayFace);

        saveLabels();
        retrainModel();

        callback.accept("✅ Sucesso: " + personName + " registrado com email " + email + ", nível " + role);
    }

    public void authenticate(Mat face, Consumer<String> callback) {
        if (idToNameMap.isEmpty()) {
            callback.accept("❌ Erro: Nenhum rosto registrado no sistema!");
            return;
        }

        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);

        Mat grayFace = new Mat();
        cvtColor(face, grayFace, COLOR_BGR2GRAY);
        faceRecognizer.predict(grayFace, label, confidence);

        int predictedLabel = label.get(0);
        double conf = confidence.get(0);

        // Limiar de reconhecimento aumentado
        double LIMIAR_RECONHECIMENTO = 60.0;

        if (predictedLabel == -1 || conf > LIMIAR_RECONHECIMENTO) {
            callback.accept("❌ Rosto não reconhecido (confiança: " + conf + ")");
        } else {
            String personName = idToNameMap.get(predictedLabel);
            String email = idToEmailMap.get(predictedLabel);
            callback.accept("✅ Autenticado como: " + personName + " (" + email + ") - Confiança: " + conf);
        }
    }

    private void retrainModel() {
        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (Map.Entry<Integer, String> entry : idToNameMap.entrySet()) {
            int id = entry.getKey();
            File personDir = new File(FACES_DIR, String.valueOf(id));
            if (!personDir.exists())
                continue;

            for (File imgFile : Objects.requireNonNull(personDir.listFiles())) {
                Mat img = opencv_imgcodecs.imread(imgFile.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);
                if (!img.empty()) {
                    images.add(img);
                    labels.add(id);
                }
            }
        }

        if (!images.isEmpty()) {
            MatVector matImages = new MatVector(images.size());
            Mat labelsMat = new Mat(labels.size(), 1, CV_32SC1);

            for (int i = 0; i < images.size(); i++) {
                matImages.put(i, images.get(i));
                labelsMat.ptr(i).putInt(labels.get(i));
            }

            faceRecognizer.train(matImages, labelsMat);
            System.out.println(
                    "Modelo treinado com " + images.size() + " imagens de " + idToNameMap.size() + " usuários");
        }
    }

    private void loadLabels() {
        File file = new File(LABELS_FILE);
        if (!file.exists())
            return;

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(";");
                if (parts.length == 3) {
                    int id = Integer.parseInt(parts[0]);
                    String name = parts[1];
                    String email = parts[2];
                    idToNameMap.put(id, name);
                    idToEmailMap.put(id, email);
                    if (id >= nextId)
                        nextId = id + 1;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void saveLabels() {
        try (PrintWriter pw = new PrintWriter(new FileWriter(LABELS_FILE))) {
            for (Map.Entry<Integer, String> entry : idToNameMap.entrySet()) {
                int id = entry.getKey();
                String name = entry.getValue();
                String email = idToEmailMap.get(id);
                pw.println(id + ";" + name + ";" + email);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}