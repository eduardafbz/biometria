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
            if (user.getFaceId() != null && userRepository.existsByFaceId(user.getFaceId())) {
                throw new RuntimeException("Rosto já cadastrado no sistema");
            }
            return userRepository.save(user);
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
    }