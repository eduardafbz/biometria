package com.unip.controller;

import com.unip.model.Role;
import com.unip.model.User;
import com.unip.service.FaceService;
import com.unip.service.UserService;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

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
    public ResponseEntity<Map<String, Object>> registerFace(@RequestParam("image") MultipartFile image, @RequestParam("name") String name, @RequestParam("email") String email, @RequestParam("role") Role role) {

        Map<String, Object> response = new HashMap<>();

        try {
            Mat matImage = convertMultipartFileToMat(image);
            CompletableFuture<String> future = new CompletableFuture<>();

            faceService.register(matImage, name, email, role, message -> {
                future.complete(message);
            });

            String result = future.get(30, TimeUnit.SECONDS);

            response.put("success", result.contains("✅"));
            response.put("message", result);
            response.put("user", Map.of("name", name, "email", email, "role", role));
            
            return ResponseEntity.ok(response);

        } catch (IOException e) {
            response.put("success", false);
            response.put("message", "Erro ao processar imagem");
            return ResponseEntity.badRequest().body(response);
        } catch (TimeoutException e) {
            response.put("success", false);
            response.put("message", "Timeout - processamento muito longo");
            return ResponseEntity.badRequest().body(response);
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", "Erro interno: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }

    @PostMapping("/authenticate")
    public ResponseEntity<Map<String, Object>> authenticatedFace (@RequestParam("image") MultipartFile image) {
        Map<String, Object> response = new HashMap<>();
        
        try {
            Mat matImage = convertMultipartFileToMat(image);
            final String[] resultMessage = new String[1];
            faceService.authenticate(matImage, message -> {
                resultMessage[0] = message;
            });
            response.put("success", resultMessage[0] != null && resultMessage[0].contains("✅"));
            response.put("message", resultMessage[0]);
            return ResponseEntity.ok(response);
        } catch (IOException e) {
            response.put("success", false);
            response.put("message", "Erro ao processar imagem: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", "Erro interno: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/list")
    public ResponseEntity<List<User>> listFaces() {
        List<User> faces = userService.findAll();
        return ResponseEntity.ok(faces);
    }
}
