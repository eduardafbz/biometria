package com.unip.application;

import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

import com.unip.util.SpringFxmlLoader;

import javafx.application.Application;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

@SpringBootApplication(scanBasePackages = "com.unip")
@EnableJpaRepositories(basePackages = "com.unip.repository")
@EntityScan(basePackages = "com.unip.model")
public class MainApp extends Application {
    private ConfigurableApplicationContext springContext;

    @Override
    public void init() {
        springContext = new SpringApplicationBuilder(MainApp.class)
                .run();
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        SpringFxmlLoader loader = new SpringFxmlLoader(springContext);
        Parent root = loader.load("/view/CameraView.fxml");

        primaryStage.setScene(new Scene(root));
        primaryStage.show();
    }

    @Override
    public void stop() {
        springContext.close();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
