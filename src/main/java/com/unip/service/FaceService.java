package com.unip.service;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.springframework.stereotype.Service;

import com.unip.model.Role;
import com.unip.model.User;

@Service
public class FaceService {
    private final FaceRecognizer faceRecognizer;
    private final Map<Integer, String> idToNameMap = new HashMap<>();
    private final Map<Integer, String> idToEmailMap = new HashMap<>();
    private final Map<Integer, Role> idToRoleMap = new HashMap<>();

    private final String FACES_DIR = "faces";

    private final Map<Integer, Integer> recognitionAttempts = new HashMap<>();
    private final Map<Integer, Integer> successfulRecognitions = new HashMap<>();

    private final FaceProcessingHelper helper;

    public FaceService() {
        this.faceRecognizer = LBPHFaceRecognizer.create();
        this.helper = new FaceProcessingHelper(FACES_DIR, faceRecognizer,
                idToNameMap, idToEmailMap, idToRoleMap, recognitionAttempts, successfulRecognitions);
    }

    public void register(Mat face, User user, BiConsumer<String, Role> callback) {
        helper.register(face, user, callback);
    }

    public Boolean authenticate(Mat face, BiConsumer<String, Role> callback) {
        return helper.authenticate(face, callback);
    }

    public void loadUsers(List<User> users) {
        helper.loadLabels(users);
        helper.retrainModel();
    }

    public void detectFaces(Mat frame, boolean drawRects) {
        helper.detectFaces(frame, drawRects);
    }

    public int countFacesInImage(Mat image) {
        return helper.countFacesInImage(image);
    }

    public void delete(User user) throws IOException {
        helper.deleteDirectory(user.getId().intValue());
    }
}