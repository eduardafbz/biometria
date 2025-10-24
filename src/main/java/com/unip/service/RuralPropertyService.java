package com.unip.service;

import java.time.LocalDate;
import java.util.List;

import com.unip.model.RuralProperty;

public interface RuralPropertyService {
    RuralProperty cadastrarNovaPropriedade(RuralProperty propriedade);

    List<RuralProperty> listarTodasPropriedades();

    RuralProperty buscarPorId (Long id);

    void deletarPropriedade(Long id);

    void updateDate(Long id, LocalDate data);

    RuralProperty atualizarPropriedade(RuralProperty property);
}
