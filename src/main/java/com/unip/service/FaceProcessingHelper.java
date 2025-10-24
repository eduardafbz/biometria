package com.unip.service;

import static org.bytedeco.opencv.global.opencv_core.CV_32FC1;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.equalizeHist;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.global.opencv_imgproc;

import com.unip.model.Role;
import com.unip.model.User;

class FaceProcessingHelper {
    private final String FACES_DIR;
    private final FaceRecognizer faceRecognizer;
    private final Map<Integer, String> idToNameMap;
    private final Map<Integer, String> idToEmailMap;
    private final Map<Integer, Role> idToRoleMap;
    private final Map<Integer, Integer> recognitionAttempts;
    private final Map<Integer, Integer> successfulRecognitions;

    private static final double LIMIAR_DUPLICATA = 45.0;
    private static final double LIMIAR_RECONHECIMENTO = 55.0;
    private static final int MIN_IMAGES_PER_USER = 3;

    FaceProcessingHelper(String facesDir,
            FaceRecognizer faceRecognizer,
            Map<Integer, String> idToNameMap,
            Map<Integer, String> idToEmailMap,
            Map<Integer, Role> idToRoleMap,
            Map<Integer, Integer> recognitionAttempts,
            Map<Integer, Integer> successfulRecognitions) {

        this.FACES_DIR = facesDir;
        this.faceRecognizer = faceRecognizer;
        this.idToNameMap = idToNameMap;
        this.idToEmailMap = idToEmailMap;
        this.idToRoleMap = idToRoleMap;
        this.recognitionAttempts = recognitionAttempts;
        this.successfulRecognitions = successfulRecognitions;
    }

    void register(Mat face, User user, BiConsumer<String, Role> callback) {
        Mat processedFace = preprocessFace(face);

        if (!idToNameMap.isEmpty()) {
            try {
                IntPointer label = new IntPointer(1);
                DoublePointer confidence = new DoublePointer(1);
                faceRecognizer.predict(processedFace, label, confidence);

                double conf = confidence.get(0);
                System.out.println("DEBUG REGISTER - Confiança: " + conf + ", Limiar: " + LIMIAR_DUPLICATA);

                if (conf < LIMIAR_DUPLICATA) {
                    String existingName = idToNameMap.get(label.get(0));
                    callback.accept(" Erro: Rosto já registrado como '" + existingName + "'", null);
                    return;
                }
            } catch (Exception e) {
                System.out.println("Modelo não treinado, primeiro registro...");
            }
        }

        if (idToEmailMap.containsValue(user.getEmail())) {
            callback.accept("Erro: Email já registrado", null);
            return;
        }

        int personId = user.getId().intValue();

        idToNameMap.put(personId, user.getName());
        idToEmailMap.put(personId, user.getEmail());
        idToRoleMap.put(personId, user.getRole());

        File personDir = new File(FACES_DIR, String.valueOf(personId));
        personDir.mkdirs();

        File[] existingImages = personDir.listFiles();
        int currentImageCount = existingImages != null ? existingImages.length : 0;

        String filename = new File(personDir, "original_" + System.currentTimeMillis() + ".png").getAbsolutePath();
        opencv_imgcodecs.imwrite(filename, processedFace);

        if (currentImageCount < 5) {
            generateAugmentedImages(processedFace, personDir);
        }

        retrainModel();

        recognitionAttempts.put(personId, 0);
        successfulRecognitions.put(personId, 0);

        callback.accept(
                "Sucesso: " + user.getName() + " registrado com email " + user.getEmail() + ", nível " + user.getRole(),
                user.getRole());
    }

    public Boolean authenticate(Mat face, BiConsumer<String, Role> callback) {
        boolean encontrado = false;

        if (idToNameMap.isEmpty()) {
            callback.accept("Erro: Nenhum rosto registrado no sistema!", null);
            return encontrado;
        }

        Mat processedFace = preprocessFace(face);
        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);

        faceRecognizer.predict(processedFace, label, confidence);

        int predictedLabel = label.get(0);
        double conf = confidence.get(0);

        if (predictedLabel == -1 || conf > LIMIAR_RECONHECIMENTO || !isConfidenceReliable(conf, predictedLabel)) {
            callback.accept("Rosto não reconhecido (confiança: " + String.format("%.2f", conf) + ")", null);
        } else {
            encontrado = true;

            recognitionAttempts.put(predictedLabel,
                    recognitionAttempts.getOrDefault(predictedLabel, 0) + 1);
            successfulRecognitions.put(predictedLabel,
                    successfulRecognitions.getOrDefault(predictedLabel, 0) + 1);

            String personName = idToNameMap.get(predictedLabel);
            String email = idToEmailMap.get(predictedLabel);
            Role role = idToRoleMap.get(predictedLabel);

            System.out.printf("RECONHECIMENTO - User: %s, Confiança: %.2f", personName, conf);

            callback.accept("Autenticado como: " + personName + " (" + email + ") - Nível: " + role +
                    " - Confiança: " + String.format("%.2f", conf), role);
        }

        return encontrado;
    }

    public void detectFaces(Mat frame, boolean drawRects) {
        try {
            CascadeClassifier classifier = loadCascade("/haarcascade_frontalface_default.xml");
            Mat gray = new Mat();
            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);

            RectVector faces = new RectVector();
            classifier.detectMultiScale(gray, faces);

            if (drawRects) {
                for (int i = 0; i < faces.size(); i++) {
                    Rect r = faces.get(i);
                    opencv_imgproc.rectangle(frame,
                            new org.bytedeco.opencv.opencv_core.Point(r.x(), r.y()),
                            new org.bytedeco.opencv.opencv_core.Point(r.x() + r.width(), r.y() + r.height()),
                            new org.bytedeco.opencv.opencv_core.Scalar(0, 255, 0, 0), 2, 0, 0);
                }
            }

        } catch (Exception e) {
            System.err.println("Erro na detecção de faces: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private CascadeClassifier loadCascade(String cascadePath) {
        try {
            CascadeClassifier classifier = new CascadeClassifier();
            String path = getClass().getResource(cascadePath).getPath();

            if (!classifier.load(path)) {
                throw new RuntimeException("Não foi possível carregar o classificador: " + cascadePath);
            }
            return classifier;
        } catch (Exception e) {
            throw new RuntimeException("Erro ao carregar classificador Haar Cascade: " + cascadePath, e);
        }
    }

    Mat preprocessFace(Mat face) {
        Mat processed = new Mat();
        Mat grayFace = new Mat();
        cvtColor(face, grayFace, COLOR_BGR2GRAY);
        equalizeHist(grayFace, processed);
        resize(processed, processed, new Size(200, 200));
        GaussianBlur(processed, processed, new Size(3, 3), 0);
        return processed;
    }

    void generateAugmentedImages(Mat original, File outputDir) {
        try {
            Mat brighter = adjustBrightness(original, 1.3);
            opencv_imgcodecs.imwrite(
                    new File(outputDir, "bright_" + System.currentTimeMillis() + ".png").getAbsolutePath(), brighter);

            Mat darker = adjustBrightness(original, 0.7);
            opencv_imgcodecs.imwrite(
                    new File(outputDir, "dark_" + System.currentTimeMillis() + ".png").getAbsolutePath(), darker);

            Mat flipped = new Mat();
            org.bytedeco.opencv.global.opencv_core.flip(original, flipped, 1);
            opencv_imgcodecs.imwrite(
                    new File(outputDir, "flipped_" + System.currentTimeMillis() + ".png").getAbsolutePath(), flipped);

            Mat rotated2 = rotateImageSimple(original, 2);
            if (!rotated2.empty()) {
                opencv_imgcodecs.imwrite(
                        new File(outputDir, "rotated2_" + System.currentTimeMillis() + ".png").getAbsolutePath(),
                        rotated2);
            }

            Mat rotatedMinus2 = rotateImageSimple(original, -2);
            if (!rotatedMinus2.empty()) {
                opencv_imgcodecs.imwrite(
                        new File(outputDir, "rotatedMinus2_" + System.currentTimeMillis() + ".png").getAbsolutePath(),
                        rotatedMinus2);
            }

        } catch (Exception e) {
            System.out.println("Erro ao gerar imagens aumentadas: " + e.getMessage());
        }
    }

    Mat adjustBrightness(Mat image, double factor) {
        Mat result = new Mat();
        image.convertTo(result, -1, factor, 0);
        return result;
    }

    Mat rotateImageSimple(Mat src, double angle) {
        try {
            Mat dst = new Mat();
            if (angle == 0)
                return src.clone();

            double radians = Math.toRadians(angle);
            double shear = Math.tan(radians);

            Mat warpMat = new Mat(2, 3, CV_32FC1);
            float[] data = {
                    1.0f, (float) shear, (float) (-shear * src.rows() / 2),
                    (float) -shear, 1.0f, (float) (shear * src.cols() / 2)
            };
            warpMat.data().put(new FloatPointer(data));
            warpAffine(src, dst, warpMat, src.size());
            return dst;

        } catch (Exception e) {
            System.out.println("Erro na rotação simples: " + e.getMessage());
            return src.clone();
        }
    }

    boolean isConfidenceReliable(double confidence, int label) {
        File userDir = new File(FACES_DIR, String.valueOf(label));
        if (!userDir.exists())
            return false;

        File[] imageFiles = userDir.listFiles();
        int imageCount = imageFiles != null ? imageFiles.length : 0;

        if (imageCount < MIN_IMAGES_PER_USER) {
            return confidence < LIMIAR_RECONHECIMENTO * 0.7;
        }

        return true;
    }

    void retrainModel() {
        try {
            List<Mat> images = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();

            for (Map.Entry<Integer, String> entry : idToNameMap.entrySet()) {
                int id = entry.getKey();
                File personDir = new File(FACES_DIR, String.valueOf(id));
                if (!personDir.exists())
                    continue;

                File[] imageFiles = personDir.listFiles();
                if (imageFiles == null || imageFiles.length < MIN_IMAGES_PER_USER) {
                    System.out.println("Usuário " + entry.getValue() + " ignorado no treinamento - poucas imagens: " +
                            (imageFiles != null ? imageFiles.length : 0));
                    continue;
                }

                for (File imgFile : imageFiles) {
                    if (imageFiles.length >= 8 &&
                            (imgFile.getName().contains("bright_") ||
                                    imgFile.getName().contains("dark_") ||
                                    imgFile.getName().contains("flipped_") ||
                                    imgFile.getName().contains("rotated"))) {
                        continue;
                    }

                    Mat img = opencv_imgcodecs.imread(imgFile.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);
                    if (!img.empty()) {
                        images.add(img);
                        labels.add(id);
                    }
                }
            }

            if (images.size() >= MIN_IMAGES_PER_USER) {
                MatVector matImages = new MatVector(images.size());
                Mat labelsMat = new Mat(labels.size(), 1, CV_32SC1);

                for (int i = 0; i < images.size(); i++) {
                    matImages.put(i, images.get(i));
                    labelsMat.ptr(i).putInt(labels.get(i));
                }

                faceRecognizer.train(matImages, labelsMat);
                System.out.println("Modelo treinado com " + images.size() + " imagens de " +
                        new HashSet<>(labels).size() + " usuários");
            } else {
                System.out.println("Dados insuficientes para treinamento: " + images.size() + " imagens");
            }
        } catch (Exception e) {
            System.err.println(" Erro no treinamento: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public void loadLabels(List<User> users) {
        for (User user : users) {
            idToNameMap.put(user.getId().intValue(), user.getName());
            idToEmailMap.put(user.getId().intValue(), user.getEmail());
            idToRoleMap.put(user.getId().intValue(), user.getRole());

            recognitionAttempts.put(user.getId().intValue(), 0);
            successfulRecognitions.put(user.getId().intValue(), 0);
        }
    }

    public int countFacesInImage(Mat image) {
        try {
            CascadeClassifier classifier = loadCascade("/haarcascade_frontalface_default.xml");
            Mat gray = new Mat();
            opencv_imgproc.cvtColor(image, gray, opencv_imgproc.COLOR_BGR2GRAY);

            RectVector faces = new RectVector();
            classifier.detectMultiScale(gray, faces);

            return (int) faces.size();
        } catch (Exception e) {
            System.err.println("Erro ao contar faces: " + e.getMessage());
            return 0;
        }
    }

    public void deleteDirectory(int id) throws IOException {
        Path path = Paths.get(FACES_DIR + "/" + id);

        if (!Files.exists(path))
            return;

        Files.walkFileTree(path, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.delete(dir);
                return FileVisitResult.CONTINUE;
            }
        });
    }
}