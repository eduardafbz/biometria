package com.unip.controller;

import java.util.Optional;

import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import com.unip.config.SpringContext;
import com.unip.model.Role;
import com.unip.model.User;
import com.unip.service.CameraService;
import com.unip.service.FaceService;
import com.unip.service.RuralPropertyService;
import com.unip.service.UserService;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;

@Component
public class UIController {

    @Autowired
    private RuralPropertyService propertyService;

    @Autowired
    private UserService userService;

    @FXML
    private Button cameraToggle;

    @FXML
    private Button registerFaceButton;

    @FXML
    private Button authFaceButton;

    @FXML
    private Button markFacesButton;

    @FXML
    private ImageView cameraView;

    private volatile boolean markFaces = false;

    @Autowired
    private CameraService cameraService;

    @Autowired
    private FaceService faceService;

    private static final String CAMERA_ON_TEXT = "Ligar Câmera";
    private static final String CAMERA_OFF_TEXT = "Desligar Câmera";

    @FXML
    private void initialize() {
        cameraToggle.setText(CAMERA_ON_TEXT);
        cameraToggle.setOnAction(e -> toggleCamera());

        var users = userService.findAll();
        faceService.loadUsers(users);

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
                    faceService.authenticate(frame, (message, role) -> {
                        if (role != null) {
                            openRoleWindow(role);
                        } else {
                            showMessage(message);
                        }
                    });
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        });

        markFacesButton.setOnAction(e -> {
            markFaces = !markFaces;
            String buttonText = markFaces ? "Parar detecção" : "Detectar rosto";
            markFacesButton.setText(buttonText);
        });

        Platform.runLater(this::setupKeyboardShortcuts);
    }

    private void setupKeyboardShortcuts() {
        Scene scene = cameraView.getScene();
        if (scene != null) {
            scene.setOnKeyPressed(event -> {
                switch (event.getCode()) {
                    case F1 -> openRoleWindow(Role.LEVEL_1);
                    case F2 -> openRoleWindow(Role.LEVEL_2);
                    case F3 -> openRoleWindow(Role.LEVEL_3);
                }
            });
        }
    }

    private void showRegistrationDialog(Mat frame) {
        int faceCount = faceService.countFacesInImage(frame);
        if (faceCount == 0) {
            showMessage("Erro: Nenhum rosto detectado na imagem. Por favor, posicione-se melhor na câmera.");
            return;
        }
        if (faceCount > 1) {
            showMessage("Erro: A imagem contém " + faceCount + " rostos. Por favor, tire uma foto com apenas UMA pessoa.");
            return;
        }

        Dialog<User> dialog = new Dialog<>();
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

        grid.add(new Label("Nome:"), 0, 0);
        grid.add(nameField, 1, 0);
        grid.add(new Label("Email:"), 0, 1);
        grid.add(emailField, 1, 1);

        dialog.getDialogPane().setContent(grid);

        Platform.runLater(() -> nameField.requestFocus());

        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == registerButtonType) {
                return User.builder().name(nameField.getText()).email(emailField.getText()).role(Role.LEVEL_1).build();
            }

            return null;
        });

        Optional<User> result = dialog.showAndWait();

        result.ifPresent(user -> {
            String name = user.getName();
            String email = user.getEmail();
            Role role = user.getRole();

            if (name == null || name.trim().isEmpty()) {
                showMessage("Erro: Nome é obrigatório!");
                return;
            }

            if (email == null || email.trim().isEmpty() || !email.contains("@")) {
                showMessage("Erro: Email válido é obrigatório!");
                return;
            }

            try {
                user = userService.saveUser(user);

                faceService.register(frame, user, (message, registeredRole) -> {
                    if (registeredRole != null) {
                        openRoleWindow(registeredRole);
                    }
                });
            } catch (Exception e) {
                showMessage("Erro: " + e.getMessage());
            }
        });
    }

    private void toggleCamera() {
        if (cameraService.isCameraActive()) {
            cameraService.stopCamera();
            cameraToggle.setText(CAMERA_ON_TEXT);
            cameraView.setVisible(false);
        } else {
            cameraService.startCamera(frame -> {
                if (markFaces) {
                    faceService.detectFaces(frame, true);
                }
                
                Platform.runLater(() -> {
                    cameraView.setImage(matToImage(frame));
                });
            });
            cameraToggle.setText(CAMERA_OFF_TEXT);
            cameraView.setVisible(true);
        }
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

    private void openRoleWindow(Role role) {
        Platform.runLater(() -> {
            try {
                String fxmlFile = "/view/MainWindow.fxml";
                String title = "";

                switch (role) {
                    case LEVEL_1:
                        title = "Sistema - Nível 1";
                        break;

                    case LEVEL_2:
                        title = "Sistema - Nível 2";
                        break;

                    case LEVEL_3:
                        title = "Sistema - Nível 3";
                        break;

                    default:
                        title = "Sistema - Nível 1";
                }

                Stage currentStage = (Stage) cameraView.getScene().getWindow();
                currentStage.close();

                ApplicationContext context = SpringContext.getApplicationContext();
                FXMLLoader loader = new FXMLLoader(getClass().getResource(fxmlFile));
                loader.setControllerFactory(context::getBean);

                Parent root = loader.load();

                Object controller = loader.getController();

                if (controller instanceof MainWindowController) {
                    MainWindowController topController = (MainWindowController) controller;
                    topController.setPropertyService(propertyService);
                    topController.setCurrentRole(role);
                }

                Stage newStage = new Stage();
                newStage.setTitle(title);
                newStage.setScene(new Scene(root));
                newStage.show();

            } catch (Exception e) {
                e.printStackTrace();
                showMessage("Erro ao abrir a janela: " + e.getMessage());
            }
        });
    }
}