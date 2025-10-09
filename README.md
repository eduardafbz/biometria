Para rodar o projeto use o comando no terminal

`mvn clean javafx:run`

Caso não tenha banco de dados MySql configurado, mude o arquivo src/main/resources/application.properties para usar o banco H2, de:

`spring.profiles.active=mysql`

para:

`spring.profiles.active=h2`