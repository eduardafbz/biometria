package com.unip.controller;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.imageio.ImageIO;

import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.unip.model.Role;
import com.unip.model.User;
import com.unip.service.FaceService;
import com.unip.service.UserService;

@RestController
@RequestMapping("/faces")
public class FaceController {

    @Autowired
    private FaceService faceService;

    @Autowired
    private UserService userService;

    private Mat convertMultipartFileToMat(MultipartFile file) throws IOException {
        BufferedImage bufferedImage = ImageIO.read(file.getInputStream());

        if (bufferedImage == null) {
            throw new IOException("Não foi possível ler a imagem");
        }

        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        int channels = 3;

        Mat mat = new Mat(height, width, opencv_core.CV_8UC3);
        byte[] buffer = new byte[width * height * channels];

        int[] pixels = new int[width * height];
        bufferedImage.getRGB(0, 0, width, height, pixels, 0, width);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            buffer[i * 3] = (byte) ((pixel >> 16) & 0xFF);
            buffer[i * 3 + 1] = (byte) ((pixel >> 8) & 0xFF);
            buffer[i * 3 + 2] = (byte) (pixel & 0xFF);
        }

        mat.data().put(buffer);
        return mat;
    }

    @PostMapping("/register")
    public ResponseEntity<Map<String, Object>> registerFace(
            @RequestParam("image") MultipartFile image,
            @RequestParam("name") String name,
            @RequestParam("email") String email,
            @RequestParam("role") Role role) {
        try {
            User user = User.builder().name(name).email(email).role(role).build();

            Mat matImage = convertMultipartFileToMat(image);
            CompletableFuture<Map<String, Object>> future = new CompletableFuture<>();

            faceService.register(matImage, user, (message, registeredRole) -> {
                Map<String, Object> result = new HashMap<>();
                result.put("success", true);
                result.put("message", message);
                result.put("user",
                        Map.of("name", name, "email", email, "role", registeredRole != null ? registeredRole : role));
                future.complete(result);
            });
            Map<String, Object> result = future.get(10, TimeUnit.SECONDS);
            return ResponseEntity.ok(result);

        } catch (IOException e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Erro ao processar imagem: " + e.getMessage());
            return ResponseEntity.badRequest().body(errorResponse);
        } catch (TimeoutException e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Timeout - processamento muito longo");
            return ResponseEntity.badRequest().body(errorResponse);
        } catch (Exception e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Erro interno: " + e.getMessage());
            return ResponseEntity.badRequest().body(errorResponse);
        }
    }

    @PostMapping("/authenticate")
    public ResponseEntity<Map<String, Object>> authenticatedFace(
            @RequestParam("image") MultipartFile image) {
        try {
            Mat matImage = convertMultipartFileToMat(image);
            CompletableFuture<Map<String, Object>> future = new CompletableFuture<>();
            faceService.authenticate(matImage, (message, role) -> {
                Map<String, Object> result = new HashMap<>();
                result.put("success", role != null);
                result.put("message", message);
                if (role != null) {
                    result.put("role", role);
                }
                future.complete(result);
            });

            Map<String, Object> result = future.get(10, TimeUnit.SECONDS);
            return ResponseEntity.ok(result);
        } catch (IOException e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Erro ao processar imagem: " + e.getMessage());
            return ResponseEntity.badRequest().body(errorResponse);
        } catch (TimeoutException e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Timeout - processamento muito longo");
            return ResponseEntity.badRequest().body(errorResponse);
        } catch (Exception e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "Erro interno: " + e.getMessage());
            return ResponseEntity.badRequest().body(errorResponse);
        }
    }

    @GetMapping("/list")
    public ResponseEntity<List<User>> listFaces() {
        List<User> faces = userService.findAll();
        return ResponseEntity.ok(faces);
    }
}
