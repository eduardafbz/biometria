package com.unip.controller;

import java.net.URL;
import java.time.LocalDate;
import java.util.Objects;
import java.util.ResourceBundle;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.unip.model.RuralProperty;
import com.unip.service.RuralPropertyService;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.DatePicker;
import javafx.scene.control.TextField;
import javafx.stage.Stage;

@Component
public class FormPropertyController implements Initializable {

    @Autowired
    private RuralPropertyService propertyService;

    private RuralProperty propertyToEdit;
    private MainWindowController mainController;

    @FXML
    private Button btn_cancel;

    @FXML
    private Button btn_save;

    @FXML
    private DatePicker dtpicker_fiscalization;

    @FXML
    private TextField txtfield_address;

    @FXML
    private TextField txtfield_owner;

    private Boolean newProperty;

    @FXML
    void cancel_edition(javafx.scene.input.MouseEvent event) {
        closeWindow();
    }

    @FXML
    void save_new_date(javafx.scene.input.MouseEvent event) {
        saveProperty();
    }

    @Override
    public void initialize(URL url, ResourceBundle rb) {
    }

    private void toggleFields(Boolean editable) {
        txtfield_address.setEditable(editable);
        txtfield_owner.setEditable(editable);

        if (!editable) {
            txtfield_address.setStyle("-fx-background-color: #f0f0f0; -fx-text-fill: #666666;");
            txtfield_owner.setStyle("-fx-background-color: #f0f0f0; -fx-text-fill: #666666;");
        }

        dtpicker_fiscalization.setStyle("-fx-background-color: white;");
    }

    public void setPropertyToEdit(RuralProperty property) {
        this.propertyToEdit = property;

        if (Objects.nonNull(property)) {
            loadPropertyData();
            toggleFields(false);
        } else {
            toggleFields(true);
        }
    }

    public void setMainController(MainWindowController mainController) {
        this.mainController = mainController;
    }

    private void loadPropertyData() {
        if (propertyToEdit != null) {

            txtfield_owner.setText(propertyToEdit.getOwner());
            txtfield_address.setText(propertyToEdit.getAddress());

            if (propertyToEdit.getInspectionDate() != null) {
                dtpicker_fiscalization.setValue(propertyToEdit.getInspectionDate());
            }
        }
    }

    private void saveProperty() {
        if (propertyToEdit == null && !newProperty) {
            showAlert("Erro", "Nenhuma propriedade selecionada para edição.");
            return;
        }

        try {
            LocalDate newDate = dtpicker_fiscalization.getValue();
            if (newDate == null) {
                showAlert("Erro", "Selecione uma data de fiscalização válida.");
                return;
            }

            if (newProperty) {
                var property = RuralProperty.builder()
                        .owner(txtfield_owner.getText())
                        .address(txtfield_address.getText())
                        .inspectionDate(newDate)
                        .build();

                propertyService.cadastrarNovaPropriedade(property);
            } else {

                // Atualiza a data diretamente no objeto
                propertyToEdit.setInspectionDate(newDate);

                // Salva no banco de dados
                propertyService.atualizarPropriedade(propertyToEdit);
            }

            if (mainController != null) {
                mainController.refreshPropertiesTables();
            }

            showAlert("Sucesso", "Salvo com sucesso!");
            closeWindow();

        } catch (Exception e) {
            e.printStackTrace();
            showAlert("Erro", "Erro ao atualizar propriedade: " + e.getMessage());
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

    public void setNewProperty(Boolean newProperty) {
        this.newProperty = newProperty;
    }
}