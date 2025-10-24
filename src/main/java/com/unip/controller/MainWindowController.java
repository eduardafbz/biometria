package com.unip.controller;

import java.io.IOException;
import java.net.URL;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.ResourceBundle;

import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Controller;

import com.unip.config.SpringContext;
import com.unip.model.Role;
import com.unip.model.RuralProperty;
import com.unip.model.User;
import com.unip.service.CameraService;
import com.unip.service.FaceService;
import com.unip.service.RuralPropertyService;
import com.unip.service.UserService;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.TableCell;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

@Controller
public class MainWindowController implements Initializable {

    @Autowired
    private CameraService cameraService;

    @Autowired
    private FaceService faceService;

    @Autowired
    private UserService userService;

    private ObservableList<User> usersList;

    private volatile boolean markFaces = false;
    private volatile boolean cameraActive = false;

    private RuralPropertyService propertyService;

    private ObservableList<RuralProperty> propertiesList;

    @FXML
    private Button properties_btn;

    @FXML
    private Button users_btn;

    @FXML
    private Button add_user_btn;

    @FXML
    private Button add_property_btn;

    @FXML
    private VBox add_users;

    @FXML
    private Button btn_register_user;

    @FXML
    private ImageView camera_view;

    @FXML
    private VBox properties_infos;

    @FXML
    private TableView<RuralProperty> properties_table;

    @FXML
    private Button btnLigarCamera;

    @FXML
    private StackPane stc_pane_pages;

    @FXML
    private TableColumn<User, Void> tb_col_acoes;

    @FXML
    private TableColumn<RuralProperty, Void> tb_col_acoes_properties;

    @FXML
    private TableColumn<User, String> tb_col_email;

    @FXML
    private TableColumn<User, String> tb_col_level_access;

    @FXML
    private TableColumn<User, String> tb_col_username;

    @FXML
    private TableColumn<RuralProperty, LocalDate> tbl_col_fisc_date;

    @FXML
    private TableColumn<RuralProperty, String> tbl_col_owner;

    @FXML
    private TableColumn<RuralProperty, String> tbl_col_address;

    @FXML
    private VBox users_info;

    @FXML
    private TableView<User> users_table;

    private Role currentRole;

    @FXML
    void open_add_users_pane(MouseEvent event) {
        add_users.setVisible(true);
        properties_infos.setVisible(false);
        users_info.setVisible(false);
    }

    @FXML
    void openAddProperty(MouseEvent event) {
        editProperty(null, true);
    }

    @FXML
    void logout(MouseEvent event) {
        Platform.runLater(() -> {
            try {
                String fxmlFile = "/view/CameraView.fxml";

                Stage currentStage = (Stage) ((Node) event.getSource()).getScene().getWindow();
                currentStage.close();

                ApplicationContext context = SpringContext.getApplicationContext();
                FXMLLoader loader = new FXMLLoader(getClass().getResource(fxmlFile));
                loader.setControllerFactory(context::getBean);

                Parent root = loader.load();

                Stage newStage = new Stage();
                newStage.setScene(new Scene(root));
                newStage.show();

            } catch (Exception e) {
                e.printStackTrace();
                showMessage("Erro ao abrir a janela: " + e.getMessage());
            }
        });
    }

    @FXML
    void open_properties_pane(MouseEvent event) {
        add_users.setVisible(false);
        properties_infos.setVisible(true);
        users_info.setVisible(false);

        stopCamera();

        refreshPropertiesTables();
    }

    @FXML
    void toggle_camera(MouseEvent event) {
        if (cameraActive) {
            stopCamera();
            btnLigarCamera.setText("Ligar Câmera");
        } else {
            startCamera();
            btnLigarCamera.setText("Desligar Câmera");
        }
    }

    @FXML
    void toggle_mark_faces(MouseEvent event) {
        markFaces = !markFaces;
    }

    @FXML
    void open_users_pane(MouseEvent event) {
        add_users.setVisible(false);
        properties_infos.setVisible(false);
        users_info.setVisible(true);

        stopCamera();

        refreshUsersTables();
    }

    @FXML
    void register_user(MouseEvent event) {
        try {
            Mat frame = cameraService.captureFrame();
            if (frame != null) {
                showRegistrationDialog(frame);
            } else {
                showMessage("Erro: Não foi possível capturar a imagem da câmera!");
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            showMessage("Erro ao capturar imagem: " + ex.getMessage());
        }
    }

    @Override
    public void initialize(URL url, ResourceBundle rb) {

        add_users.setVisible(false);
        properties_infos.setVisible(true);
        users_info.setVisible(false);

        setupUsersTable();
        setupPropertiesTable();
        loadUsersData();
        loadPropertiesData();

        btnLigarCamera.setText("Ligar Câmera");
    }

    private void setupUsersTable() {
        tb_col_username.setCellValueFactory(new PropertyValueFactory<>("name"));
        tb_col_email.setCellValueFactory(new PropertyValueFactory<>("email"));
        tb_col_level_access.setCellValueFactory(new PropertyValueFactory<>("role"));

        setupUserActionsColumn();
    }

    private void setupPropertiesTable() {
        tbl_col_owner.setCellValueFactory(new PropertyValueFactory<>("owner"));
        tbl_col_address.setCellValueFactory(new PropertyValueFactory<>("address"));
        tbl_col_fisc_date.setCellValueFactory(new PropertyValueFactory<>("inspectionDate"));

        setupPropertiesActionsColumn();
    }

    private void setupPropertiesActionsColumn() {
        tb_col_acoes_properties.setCellFactory(param -> new TableCell<RuralProperty, Void>() {
            private final Button btnEditar = new Button("Editar");
            private final Button btnExcluir = new Button("Excluir");
            private final HBox botoes = new HBox(btnEditar, btnExcluir);

            {
                btnEditar.setStyle("-fx-background-radius: 10; -fx-font-size: 14px; -fx-cursor: hand;");
                btnExcluir.setStyle("-fx-background-radius: 10; -fx-font-size: 14px; -fx-cursor: hand;");

                btnEditar.setTooltip(new Tooltip("Editar propriedade"));
                btnExcluir.setTooltip(new Tooltip("Excluir propriedade"));

                botoes.setSpacing(8);
                botoes.setAlignment(Pos.CENTER);

                btnEditar.setOnAction(event -> {
                    RuralProperty property = getTableView().getItems().get(getIndex());
                    editProperty(property, false);
                });

                btnExcluir.setOnAction(event -> {
                    RuralProperty property = getTableView().getItems().get(getIndex());
                    deleteProperty(property);
                });
            }

            @Override
            protected void updateItem(Void item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setGraphic(null);
                } else {
                    setGraphic(botoes);
                }
            }
        });
    }

    private void editProperty(RuralProperty property, Boolean newProperty) {
        try {
            String title = newProperty ? "Cadastrar Propriedade" : "Editar Propriedade";

            ApplicationContext context = SpringContext.getApplicationContext();
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/view/FormProperty.fxml"));
            loader.setControllerFactory(context::getBean);

            Parent root = loader.load();

            FormPropertyController formController = loader.getController();
            formController.setPropertyToEdit(property);
            formController.setNewProperty(newProperty);
            formController.setMainController(this);

            Stage stage = new Stage();
            stage.setTitle(title);
            stage.setScene(new Scene(root));
            stage.initModality(Modality.APPLICATION_MODAL);
            stage.setResizable(true);
            stage.showAndWait();
        } catch (IOException e) {
            e.printStackTrace();
            showMessage("Erro ao abrir formulário de edição: " + e.getMessage());
        }
    }

    private void deleteProperty(RuralProperty property) {
        Alert confirmacao = new Alert(Alert.AlertType.CONFIRMATION);
        confirmacao.setTitle("Confirmação de Exclusão");
        confirmacao.setHeaderText("Excluir Propriedade");
        confirmacao.setContentText("Tem certeza que deseja excluir a propriedade " + property.getOwner() + "?");

        Optional<ButtonType> resultado = confirmacao.showAndWait();
        if (resultado.isPresent() && resultado.get() == ButtonType.OK) {
            try {
                propertyService.deletarPropriedade(property.getId());

                propertiesList.remove(property);

                showMessage("Propriedade excluída com sucesso!");

            } catch (Exception e) {
                e.printStackTrace();
                showMessage("Erro ao excluir propriedade: " + e.getMessage());
            }
        }
    }

    private void setupUserActionsColumn() {
        tb_col_acoes.setCellFactory(param -> new TableCell<User, Void>() {
            private final Button btnEditar = new Button("Editar");
            private final Button btnExcluir = new Button("Excluir");
            private final HBox botoes = new HBox(btnEditar, btnExcluir);

            {
                btnEditar.setStyle("-fx-background-radius: 10; -fx-font-size: 14px; -fx-cursor: hand;");
                btnExcluir.setStyle("-fx-background-radius: 10; -fx-font-size: 14px; -fx-cursor: hand;");

                btnEditar.setTooltip(new Tooltip("Editar usuário"));
                btnExcluir.setTooltip(new Tooltip("Excluir usuário"));

                botoes.setSpacing(8);
                botoes.setAlignment(Pos.CENTER);

                btnEditar.setOnAction(event -> {
                    User user = getTableView().getItems().get(getIndex());
                    editarUsuario(user);
                });

                btnExcluir.setOnAction(event -> {
                    User user = getTableView().getItems().get(getIndex());
                    excluirUsuario(user);
                });

                if (currentRole != Role.LEVEL_3) {
                    btnEditar.setVisible(false);
                    btnExcluir.setVisible(false);
                }
            }

            @Override
            protected void updateItem(Void item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setGraphic(null);
                } else {
                    setGraphic(botoes);
                }
            }
        });
    }

    private void excluirUsuario(User user) {
        Alert confirmacao = new Alert(Alert.AlertType.CONFIRMATION);
        confirmacao.setTitle("Confirmação de Exclusão");
        confirmacao.setHeaderText("Excluir Usuário");
        confirmacao.setContentText("Tem certeza que deseja excluir o usuário " + user.getName() + "?");

        Optional<ButtonType> resultado = confirmacao.showAndWait();
        if (resultado.isPresent() && resultado.get() == ButtonType.OK) {
            try {

                faceService.delete(user);
                userService.delete(user.getId());

                usersList.remove(user);

                showMessage("Usuário excluído com sucesso!");

            } catch (Exception e) {
                e.printStackTrace();
                showMessage("Erro ao excluir usuário: " + e.getMessage());
            }
        }
    }

    private void editarUsuario(User user) {
        try {
            ApplicationContext context = SpringContext.getApplicationContext();
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/view/FormUser.fxml"));
            loader.setControllerFactory(context::getBean);

            Parent root = loader.load();

            FormUserController formController = loader.getController();
            formController.setUserToEdit(user);
            formController.setMainController(this);

            Stage stage = new Stage();
            stage.setTitle("Editar Usuário - " + user.getName());
            stage.setScene(new Scene(root));
            stage.initModality(Modality.APPLICATION_MODAL);
            stage.setResizable(false);
            stage.showAndWait();

        } catch (IOException e) {
            e.printStackTrace();
            showMessage("Erro ao abrir formulário de edição: " + e.getMessage());
        }
    }

    private void loadUsersData() {

        List<User> users = userService.findAll();

        usersList = FXCollections.observableArrayList(users);

        users_table.setItems(usersList);
    }

    private void loadPropertiesData() {
        if (propertyService != null) {
            List<RuralProperty> properties = propertyService.listarTodasPropriedades();

            propertiesList = FXCollections.observableArrayList(properties);

            properties_table.setItems(propertiesList);
        } else {
            System.err.println("ERRO: propertyService é null!");
        }
    }

    public void refreshUsersTables() {
        loadUsersData();

    }

    public void refreshPropertiesTables() {
        loadPropertiesData();
    }

    private void startCamera() {
        cameraService.startCamera(frame -> {
            Platform.runLater(() -> {
                if (camera_view != null) {
                    camera_view.setImage(matToImage(frame));
                }
            });
        });
        cameraActive = true;
    }

    private void stopCamera() {
        if (cameraActive) {
            cameraService.stopCamera();
            cameraActive = false;
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

        ComboBox<Role> roleComboBox = new ComboBox<>();
        roleComboBox.getItems().addAll(Role.LEVEL_1, Role.LEVEL_2);
        roleComboBox.setValue(Role.LEVEL_1);

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
                return User.builder().name(nameField.getText()).email(emailField.getText())
                        .role(roleComboBox.getValue()).build();
            }
            return null;
        });

        Optional<User> result = dialog.showAndWait();

        result.ifPresent(user -> {
            String name = user.getName();
            String email = user.getEmail();

            if (name == null || name.trim().isEmpty()) {
                showMessage("Erro: Nome é obrigatório!");
                return;
            }

            if (email == null || email.trim().isEmpty() || !email.contains("@")) {
                showMessage("Erro: Email válido é obrigatório!");
                return;
            }

            try {
                if (!faceService.authenticate(frame, (message, role) -> {
                })) {
                    user = userService.saveUser(user);

                    faceService.register(frame, user, (message, registeredRole) -> {
                        if (registeredRole != null) {
                            showMessage(message);
                            refreshUsersTables();
                        }
                    });
                } else {
                    showMessage("Rosto já cadastrado no sistema");
                }
            } catch (Exception e) {
                showMessage("Erro: " + e.getMessage());
            }
        });
    }

    private javafx.scene.image.Image matToImage(Mat frame) {
        try {
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
        } catch (Exception e) {
            return null;
        }
    }

    private void showMessage(String message) {
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Sistema");
            alert.setHeaderText(null);
            alert.setContentText(message);
            alert.show();
        });
    }

    public void shutdown() {
        stopCamera();
    }

    public void setPropertyService(RuralPropertyService propertyService) {
        this.propertyService = propertyService;
        loadPropertiesData();
    }

    public void setUserService(UserService userService) {
        this.userService = userService;
        if (users_table != null) {
            loadUsersData();
        }
    }

    public void setCurrentRole(Role role) {
        System.out.println("\n\n\n\n\n\n\n\n\n\n\n");
        System.out.println("=======================");
        System.out.println("Role atual: " + role);

        this.currentRole = role;

        switch (role) {
            case LEVEL_2:
                users_btn.setVisible(true);
                add_user_btn.setVisible(true);
                add_property_btn.setVisible(false);
                break;

            case LEVEL_3:
                users_btn.setVisible(true);
                add_user_btn.setVisible(true);
                add_property_btn.setVisible(true);
                break;

            case LEVEL_1:
            default:
                users_btn.setVisible(false);
                add_user_btn.setVisible(false);
                add_property_btn.setVisible(false);
                break;
        }
    }
}