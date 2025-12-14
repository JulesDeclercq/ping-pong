# *PingPong AI*

<img src="pictures\robot.png" width="500">

# Voorwoord

Met dit project sluiten wij onze opdracht voor AI Programming af.

Tijdens het uitvoeren van dit project hebben wij enorm veel geleerd over kunstmatige intelligentie en de toepassingen ervan in de praktijk. Het project heeft ons niet alleen inzicht gegeven in het ontwikkelen van AI-systemen, maar ook in het analyseren en verbeteren van algoritmes.

Wij willen graag onze leerkracht, **Frankie Loret**, bedanken voor zijn begeleiding. Dankzij zijn lessen en feedback konden wij dit project op een succesvolle manier afronden.

Wij wensen u veel leesplezier bij het doorlezen van ons verslag.


# Inhoudsopgave

1. [Doelstelling(en)](#doelstellingen)  
2. [Probleemstelling](#probleemstelling)  
3. [Analyse](#analyse)  
4. [Resultaat](#resultaat)  
5. [Uitbreiding](#uitbreiding)  
6. [Conclusie](#conclusie)  
7. [Bibliografie](#bibliografie)  

---

# Doelstellingen

## Hoofddoelstelling

Het hoofddoel van dit project is om te onderzoeken hoe twee virtuele tafeltennisspelers zelfstandig kunnen leren om een pingpongwedstrijd te spelen in een gesimuleerde 3D-omgeving. Het systeem moet in staat zijn om op basis van eerdere ervaringen slagbeslissingen te verbeteren en op die manier steeds effectiever de bal terug te slaan.

## Subdoelstellingen

Om dit hoofddoel te bereiken, zijn de volgende subdoelstellingen geformuleerd:

1. **Simulatie opzetten**  
   - Creëren van een 3D-tafeltennissimulatie met een tafel, net, bal en twee paddles.  
   - Zorgen dat de fysica van het spel voldoende realistisch is om nuttige leerervaringen te genereren.

2. **Leersysteem implementeren**  
   - Ontwikkelen van een eenvoudig AI-leeralgoritme dat voor elke paddle-positie kan bepalen hoe de bal geslagen moet worden (kracht, hoek en spin).  
   - Opslaan van succesvolle slagen in CSV-bestanden zodat de spelers kunnen leren van eerdere ervaringen.

3. **Exploratie en exploitatie combineren**  
   - Implementeren van een strategie waarbij de AI zowel eerder succesvolle slagen hergebruikt (**exploitatie**) als nieuwe combinaties probeert (**exploratie**).

4. **Leerproces evalueren**  
   - Observeren hoe de AI-spelers hun prestaties verbeteren naarmate er meer data beschikbaar is.  
   - Patronen en voorkeuren van de AI in kaart brengen om inzicht te krijgen in het leerproces.



# Probleemstelling 

Tafeltennis simulatie in een 3D-omgeving is complex door de continue bewegingen van de bal en de variabele slagen van spelers.  
Het kernprobleem dat dit project adresseert is:  

- Hoe kan een AI leren effectieve slagen te kiezen zonder dat alle mogelijke scenario’s vooraf geprogrammeerd worden?  
- Hoe kan het leersysteem ervaring opbouwen en patronen herkennen om steeds betere beslissingen te nemen tijdens het spel?  

Dit vormt de basis voor het ontwerp van het leeralgoritme en de virtuele simulatie.


# Analyse 

## Simulatieomgeving

Voor dit project werd een 3D-simulatie van tafeltennis gemaakt met behulp van **PyBullet**. Deze library wordt gebruikt voor de **collision detection** en **rendering** van de omgeving. De fysica is bewust eenvoudig gehouden om het leerproces controleerbaar te maken.

De simulatie bestaat uit een tafel, een net, een bal en twee paddles. Wanneer de bal een paddle raakt, wordt het nieuwe traject berekend aan de hand van een vaste paraboolfunctie die door het raakpunt op paddle-hoogte loopt. Zo blijft het gedrag van de bal voorspelbaar en consistent.

## Slagberekening en baltraject

Elke slag wordt bepaald door drie parameters:  

- **Kracht (strength):** bepaalt hoe ver de bal vliegt door de parabool uit te rekken  
- **Hoek (angle):** roteert het traject links of rechts over de tafel  
- **Spin:** voegt een extra rotatie toe wanneer de bal op de tafel botst  

Lichte ruis wordt toegevoegd als de bal de paddle niet perfect in het midden raakt, om variatie in het spel te behouden.

## Puntentelling en spelregels

- Het systeem registreert automatisch welk punt aan welke speler wordt toegekend.  
- Een instelbare **winning_score** bepaalt hoeveel punten nodig zijn om te winnen. Een waarde van -1 laat het spel onbeperkt doorgaan.  
- Afhankelijk van het resultaat van de slag wordt het leeralgoritme voorzien van feedback over de effectiviteit van de slag.

## Leren met CSV-bestanden

- Elke speler heeft een eigen CSV-bestand waarin **paddle-posities** worden gekoppeld aan slagparameters (kracht, hoek, spin).  
- Slechts succesvolle slagen worden opgeslagen; ballen die het net raken of van de tafel vliegen worden genegeerd.  
- Bij herhaling op vergelijkbare posities worden de eerder opgeslagen waarden hergebruikt.

## Exploratie en exploitatie

- **Exploitatie:** hergebruiken van eerder succesvolle slagen  
- **Exploratie:** licht afwijkende of willekeurige waarden testen  

Aanvankelijk is 66% van de slagen exploitatie en 34% exploratie. Bij een bal die niet wordt teruggeslagen verschuift dit naar 88% exploitatie en 12% exploratie. Zo leert de AI effectievere trajecten vaker te herhalen.

## Verfijning van het leerproces

- Paddle-posities worden afgerond op één cijfer na de komma voor lichte variatie.  
- Naburige posities worden bij succes licht aangepast, waardoor het leerproces sneller gaat.  
- Het proces loopt volledig automatisch, waardoor de AI stap voor stap gerichter leert spelen.


# Resultaat 

## Ontwikkeling van prestaties

- In het begin veel willekeurige slagen, waardoor ballen vaak in het net of naast de tafel belanden.  
- Naarmate de CSV-bestanden gevuld raken, neemt het aantal succesvolle slagen toe.  
- De AI maakt gebruik van eerdere ervaringen om slagen effectiever te kiezen.

## Leercurves en voortgang

- Analyse van CSV-gegevens toont een duidelijke verbetering: minder afhankelijk van toeval en vaker gebruik van eerder geleerde combinaties.  

## Gedragspatronen van de AI

- Consistentere slagen over het net  
- Minder extreme hoeken en krachtinstellingen  
- Voorkeur voor zones waar eerdere slagen succesvol waren  


# Uitbreiding 

- Geavanceerdere algoritmes om tactisch slimmer te spelen  
- Extra feedbackmechanismen zoals voorspelling van tegenstander of strategische keuzes  
- Mogelijke koppeling met fysieke robots om virtuele kennis naar de echte wereld te vertalen  

# Conclusie 

Dit project toont dat twee virtuele tafeltennisspelers met een relatief eenvoudig leersysteem hun spel stapsgewijs kunnen verbeteren. Door bij elke slag parameters en resultaat op te slaan, ontstaat een ervaringsdatabase die toekomstige beslissingen verbetert.  

De AI leert patronen te herkennen, waardoor het spel consistenter en doelgerichter wordt. Ondanks beperkingen zoals focus op eenvoudige succescriteria, vormt dit project een solide basis voor verdere experimenten met complexere algoritmes en robotica-toepassingen.

<img src="pictures\Schermafbeelding 2025-12-14 160124.png" width="300">



# Bibliografie

1. **PyBullet Documentation.** [https://pybullet.org](https://pybullet.org)  
2. Sutton, R.S., Barto, A.G. *Reinforcement Learning: An Introduction*. MIT Press, 2018.  
3. Russell, S., Norvig, P. *Artificial Intelligence: A Modern Approach*. Pearson, 2021.  
4. Youtube [Youtube.com](https://www.youtube.com)
5. Chatgpt [Chatgpt.com](https://chatgpt.com/)

