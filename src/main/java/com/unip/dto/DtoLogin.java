package com.unip.dto;

public class DtoLogin {
    public record LoginRequest(String email, String password) {};
    public record LoginFaceRequest(String faceId) {};
}
