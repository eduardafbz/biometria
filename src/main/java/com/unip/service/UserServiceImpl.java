package com.unip.service;

import com.unip.model.User;
import com.unip.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;

    public UserServiceImpl(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public User saveUser(User user) {
        if (findByEmail(user.getEmail()).isPresent()) {
            throw new RuntimeException("Email já cadastrado no sistema");
        }

        try {
            return userRepository.save(user);
        } catch (Exception e) {
            throw e;
        }
    }

    @Override
    public User findById(Long id) {
        return userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Usuário não encontrado"));
    }

    @Override
    public Optional<User> findByEmail(String email) {
        return userRepository.findByEmail(email);
    }

    @Override
    public boolean checkFaceId(String faceId) {
        return userRepository.existsByFaceId(faceId);
    }

    @Override
    public List<User> findAll() {
        return userRepository.findAll();
    }

    @Override
    public void delete(Long id) {
        userRepository.deleteById(id);
    }

    @Override
    public User update(User user) {
        User existingUser = userRepository.findById(user.getId())
                .orElseThrow(() -> new RuntimeException("Usuário não encontrado"));

        if (user.getFaceId() != null &&
                !user.getFaceId().equals(existingUser.getFaceId()) &&
                userRepository.existsByFaceId(user.getFaceId())) {
            throw new RuntimeException("Rosto já cadastrado no sistema");
        }

        existingUser.setName(user.getName());
        existingUser.setEmail(user.getEmail());
        existingUser.setRole(user.getRole());
        existingUser.setFaceId(user.getFaceId());

        return userRepository.save(existingUser);
    }

}