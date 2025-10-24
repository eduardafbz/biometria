package com.unip.service;

import com.unip.model.User;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public interface UserService {
    User saveUser(User user);
    User findById(Long id);
    Optional<User> findByEmail(String email);
    boolean checkFaceId(String faceId);
    List<User> findAll();
    void delete(Long id);
    User update(User user);
}