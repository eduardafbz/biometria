package com.unip.controller;

import java.net.URL;
import java.util.ResourceBundle;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.unip.model.Role;
import com.unip.model.User;
import com.unip.service.UserService;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;

@Component
public class FormUserController implements Initializable {

    @Autowired
    private UserService userService;

    private User userToEdit;

    private MainWindowController mainController;

    @FXML
    private Button btn_cancel;

    @FXML
    private Button btn_save;

    @FXML
    private ComboBox<Role> combobox_level_access;

    @FXML
    private TextField txtfield_email;

    @FXML
    private TextField txtfield_username;

    @FXML
    void cancel_edition(MouseEvent event) {
        closeWindow();
    }

    @FXML
    void save_new_access_level(MouseEvent event) {
        saveUser();
    }

    @Override
    public void initialize(URL url, ResourceBundle rb) {
        setupComboBox();
        configureFields();

    }

    private void setupComboBox() {
        combobox_level_access.getItems().addAll(Role.LEVEL_1, Role.LEVEL_2, Role.LEVEL_3);

        combobox_level_access.setCellFactory(cb -> new javafx.scene.control.ListCell<>() {
            @Override
            protected void updateItem(Role role, boolean empty) {
                super.updateItem(role, empty);
                if (empty || role == null) {
                    setText(null);
                } else {
                    setText(formatRole(role));
                }
            }
        });

        combobox_level_access.setButtonCell(new javafx.scene.control.ListCell<>() {
            @Override
            protected void updateItem(Role role, boolean empty) {
                super.updateItem(role, empty);
                if (empty || role == null) {
                    setText(null);
                } else {
                    setText(formatRole(role));
                }
            }
        });
    }

    private String formatRole(Role role) {
        return switch (role) {
            case LEVEL_1 -> "Nível 1";
            case LEVEL_2 -> "Nível 2";
            case LEVEL_3 -> "Nível 3";
        };
    }

    private void configureFields() {
        txtfield_username.setEditable(false);
        txtfield_email.setEditable(false);

        txtfield_username.setStyle("-fx-background-color: #f0f0f0;");
        txtfield_email.setStyle("-fx-background-color: #f0f0f0;");
    }

    public void setUserToEdit(User user) {
        this.userToEdit = user;
        loadUserData();
    }

    public void setMainController(MainWindowController mainController) {
        this.mainController = mainController;
    }

    private void loadUserData() {
        if (userToEdit != null) {
            txtfield_username.setText(userToEdit.getName());
            txtfield_email.setText(userToEdit.getEmail());
            combobox_level_access.setValue(userToEdit.getRole());
        }
    }

    private void saveUser() {
        if (userToEdit == null) {
            showAlert("Erro", "Nenhum usuário selecionado para edição.");
            return;
        }

        try {
            Role newRole = combobox_level_access.getValue();
            if (newRole == null) {
                showAlert("Erro", "Selecione um nível de acesso válido.");
                return;
            }

            userToEdit.setRole(newRole);
            userService.update(userToEdit);

            if (mainController != null) {
                mainController.refreshUsersTables();
            }

            showAlert("Sucesso", "Nível de acesso atualizado com sucesso!");
            closeWindow();

        } catch (Exception e) {
            e.printStackTrace();
            showAlert("Erro", "Erro ao atualizar usuário: " + e.getMessage());
        }
    }

    private void closeWindow() {
        Stage stage = (Stage) btn_cancel.getScene().getWindow();
        stage.close();
    }

    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
