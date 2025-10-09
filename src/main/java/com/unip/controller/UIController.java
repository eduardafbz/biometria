package com.unip.controller;

import com.unip.model.Role;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.image.ImageView;

import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.unip.service.CameraService;
import com.unip.service.FaceService;

import java.util.Optional;

@Component
public class UIController {

    @FXML
    private RadioButton cameraToggle;

    @FXML
    private RadioButton markFacesToggle;

    @FXML
    private Button registerFaceButton;

    @FXML
    private Button authFaceButton;

    @FXML
    private ImageView cameraView;

    private volatile boolean markFaces = false;

    @Autowired
    private CameraService cameraService;

    @Autowired
    private FaceService faceService;

    private static final String CAMERA_ON_TEXT = "Turn On Camera";
    private static final String CAMERA_OFF_TEXT = "Turn Off Camera";

    // Classe interna para armazenar nome, email e role
    private static class UserData {
        private final String name;
        private final String email;
        private final Role role;

        public UserData(String name, String email, Role role) {
            this.name = name;
            this.email = email;
            this.role = role;
        }

        public String getName() {
            return name;
        }

        public String getEmail() {
            return email;
        }

        public Role getRole() {
            return role;
        }
    }

    @FXML
    private void initialize() {
        cameraToggle.setText(CAMERA_ON_TEXT);
        cameraToggle.setOnAction(e -> toggleCamera());
        markFacesToggle.setOnAction(e -> markFaces = !markFaces);

        registerFaceButton.setOnAction(e -> {
            try {
                Mat frame = cameraService.captureFrame();
                if (frame != null) {
                    showRegistrationDialog(frame);
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        });

        authFaceButton.setOnAction(e -> {
            try {
                Mat frame = cameraService.captureFrame();
                if (frame != null) {
                    faceService.authenticate(frame, this::showMessage);
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        });
    }

    private void showRegistrationDialog(Mat frame) {
        Dialog<UserData> dialog = new Dialog<>();
        dialog.setTitle("Registrar Novo Usuário");
        dialog.setHeaderText("Digite os dados do usuário:");

        ButtonType registerButtonType = new ButtonType("Registrar");
        dialog.getDialogPane().getButtonTypes().addAll(registerButtonType, ButtonType.CANCEL);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);

        TextField nameField = new TextField();
        nameField.setPromptText("Nome completo");
        TextField emailField = new TextField();
        emailField.setPromptText("email@exemplo.com");

        ComboBox<Role> roleComboBox = new ComboBox<>();
        roleComboBox.getItems().addAll(Role.values());
        roleComboBox.setValue(Role.LEVEL_1); // Valor padrão
        roleComboBox.setPromptText("Selecione o nível");

        grid.add(new Label("Nome:"), 0, 0);
        grid.add(nameField, 1, 0);
        grid.add(new Label("Email:"), 0, 1);
        grid.add(emailField, 1, 1);
        grid.add(new Label("Nível:"), 0, 2);
        grid.add(roleComboBox, 1, 2);

        dialog.getDialogPane().setContent(grid);

        Platform.runLater(() -> nameField.requestFocus());

        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == registerButtonType) {
                return new UserData(nameField.getText(), emailField.getText(), roleComboBox.getValue());
            }
            return null;
        });

        Optional<UserData> result = dialog.showAndWait();

        result.ifPresent(userData -> {
            String name = userData.getName();
            String email = userData.getEmail();
            Role role = userData.getRole();

            if (name == null || name.trim().isEmpty()) {
                showMessage("Erro: Nome é obrigatório!");
                return;
            }

            if (email == null || email.trim().isEmpty() || !email.contains("@")) {
                showMessage("Erro: Email válido é obrigatório!");
                return;
            }

            faceService.register(frame, name, email, role, this::showMessage);
        });
    }

    private void toggleCamera() {
        if (cameraService.isCameraActive()) {
            cameraService.stopCamera();
            cameraToggle.setText(CAMERA_ON_TEXT);
        } else {
            cameraService.startCamera(frame -> {
                Platform.runLater(() -> {
                    cameraView.setImage(matToImage(frame));
                });
            });
            cameraToggle.setText(CAMERA_OFF_TEXT);
        }
        markFacesToggle.setVisible(cameraService.isCameraActive());
    }

    private javafx.scene.image.Image matToImage(Mat frame) {
        int width = frame.cols();
        int height = frame.rows();
        int channels = frame.channels();
        byte[] buffer = new byte[width * height * channels];
        frame.data().get(buffer);

        javafx.scene.image.WritableImage image = new javafx.scene.image.WritableImage(width, height);
        javafx.scene.image.PixelWriter pw = image.getPixelWriter();
        pw.setPixels(0, 0, width, height, javafx.scene.image.PixelFormat.getByteRgbInstance(), buffer, 0,
                width * channels);
        return image;
    }

    private void showMessage(String message) {
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Face Recognition");
            alert.setHeaderText(null);
            alert.setContentText(message);
            alert.show();
        });
    }

    public void shutdown() {
        cameraService.stopCamera();
    }
}