package com.unip.repository;

import java.time.LocalDate;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.unip.model.RuralProperty;

import jakarta.transaction.Transactional;

@Repository
public interface RuralPropertyRepository extends JpaRepository<RuralProperty, Long> {
    void deleteById(Long id);

    @Modifying
    @Transactional
    @Query("UPDATE RuralProperty r SET r.inspectionDate = :date WHERE r.id = :id")
    void updateInspectionDate(@Param("id") Long id, @Param("date") LocalDate date);
}
