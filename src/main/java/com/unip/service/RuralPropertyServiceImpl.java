package com.unip.service;

import java.time.LocalDate;
import java.util.List;

import org.springframework.stereotype.Service;

import com.unip.model.RuralProperty;
import com.unip.repository.RuralPropertyRepository;

@Service
public class RuralPropertyServiceImpl implements RuralPropertyService {

    private final RuralPropertyRepository repository;

    public RuralPropertyServiceImpl(RuralPropertyRepository repository) {
        this.repository = repository;
    }

    @Override
    public RuralProperty cadastrarNovaPropriedade(RuralProperty property) {
        return repository.save(property);
    }

    @Override
    public List<RuralProperty> listarTodasPropriedades() {
        return repository.findAll();
    }

    @Override
    public void deletarPropriedade(Long id) {
        repository.deleteById(id);
    }

    @Override
    public void updateDate(Long id, LocalDate date) {
        repository.updateInspectionDate(id, date);
    }

    @Override
    public RuralProperty buscarPorId(Long id) {
        return repository.findById(id)
                .orElseThrow(() -> new RuntimeException("Usuário não encontrado"));
    }

    @Override
    public RuralProperty atualizarPropriedade(RuralProperty property) {

        RuralProperty existingProperty = buscarPorId(property.getId());
        
        existingProperty.setOwner(property.getOwner());
        existingProperty.setInspectionDate(property.getInspectionDate());
        
        return repository.save(existingProperty);
    }
    

}
