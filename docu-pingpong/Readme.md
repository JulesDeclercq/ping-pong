# *Ping-Pong*

<img src="pictures\robot.png" width="500">

## Voorwoord

Met dit project sluiten wij onze opdracht voor AI Programming af.  

Tijdens het uitvoeren van dit project hebben wij enorm veel geleerd over kunstmatige intelligentie en de toepassingen ervan in de praktijk. Het project gaf ons inzicht in het ontwikkelen van AI-systemen, het analyseren van algoritmes en het verbeteren van leersystemen in een 3D-simulatieomgeving.  

Wij willen graag onze leerkracht, **Frankie Loret**, bedanken voor zijn begeleiding en feedback. Dankzij zijn lessen en steun konden wij dit project op een succesvolle manier afronden.  

Wij wensen u veel leesplezier bij het doorlezen van ons verslag.


##  Inhoudsopgave

1. [Doelstelling(en)](#doelstellingen)  
2. [Probleemstelling](#probleemstelling)  
3. [Analyse](#analyse)  
4. [Resultaat](#resultaat)  
5. [Uitbreiding](#uitbreiding)  
6. [Conclusie](#conclusie)  
7. [Bibliografie](#bibliografie)  


##  Doelstellingen

###  Hoofddoelstelling

Het hoofddoel van dit project is om te onderzoeken hoe twee virtuele tafeltennisspelers zelfstandig kunnen leren om een pingpongwedstrijd te spelen in een gesimuleerde 3D-omgeving. Het systeem moet in staat zijn om op basis van eerdere ervaringen slagbeslissingen te verbeteren en op die manier steeds effectiever de bal terug te slaan.

###  Subdoelstellingen

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
   - 
# Probleemstelling

Het kernprobleem van dit project is het ontwikkelen van een AI-systeem dat zelfstandig kan leren tafeltennis te spelen in een gesimuleerde 3D-omgeving, zonder dat elke mogelijke slag vooraf geprogrammeerd is.  

Tafeltennis is een dynamisch spel waarin elke slag afhankelijk is van meerdere variabelen zoals paddle-positie, kracht, hoek en spin van de bal. Het is daarom praktisch onmogelijk om alle mogelijke scenario’s van tevoren vast te leggen. Het doel is een AI die patronen herkent, leert van ervaring en beslissingen maakt die de kans op een succesvolle slag maximaliseren.  

Daarnaast is het van belang dat de AI leert omgaan met variatie en imperfecties, zoals het raken van de bal niet exact in het midden van de paddle, waardoor kleine afwijkingen in trajecten ontstaan. Deze uitdaging maakt het project relevant in de context van **robotica en autonome systemen**, waar machines continu leren van een dynamische omgeving en hun beslissingen verbeteren zonder menselijke tussenkomst.  

**Doelgroep:**  
- Studenten en onderzoekers in AI en robotica.  
- Ontwikkelaars die geïnteresseerd zijn in reinforcement learning en simulatiegedreven AI.  
- Onderwijsprojecten waarbij simulatie en adaptief leren centraal staan.  

Door dit project aan te pakken, kunnen we beter begrijpen hoe simpele leeralgoritmes in een complexe, continue 3D-omgeving nuttige beslissingen kunnen nemen.

## Analyse

### Stappen in de analyse van de probleemstelling

Om het probleem van een AI-tafeltennisspelend systeem aan te pakken, zijn we begonnen met het opzetten van een **simulatieomgeving** waarin virtuele spelers kunnen leren. Hiervoor hebben we gebruikgemaakt van **PyBullet**, een 3D-simulatie- en physics-engine die collision detection en rendering ondersteunt. De fysica is bewust vereenvoudigd: de balbeweging wordt berekend met een **paraboolfunctie** gebaseerd op drie parameters – kracht (strength), hoek (angle) en spin – en het raakpunt van de paddle. Het traject gaat altijd door het paddle-raakpunt, zodat het gedrag voorspelbaar blijft maar dynamisch genoeg is voor zinvol leren.

Vervolgens hebben we het **AI-leersysteem** ontwikkeld dat voor elke x-y-paddlepositie bepaalt welke slagparameters gebruikt worden. Succesvolle slagen worden opgeslagen in CSV-bestanden, zodat de spelers in toekomstige rally’s deze ervaring kunnen hergebruiken. Het algoritme combineert **exploitatie van eerder succesvolle slagen** en **exploratie van nieuwe parameters**, waarbij de verhouding adaptief wordt aangepast afhankelijk van het resultaat van de tegenstander.

---

### Vergelijkbare projecten

Er bestaan enkele projecten die AI leren tafeltennissen of soortgelijke 3D-games aanpakken, zoals:

- **OpenAI Gym Table Tennis simulaties**: richt zich op reinforcement learning in een virtuele pingpong-omgeving.  
- **DeepMind Control Suite**: gebruikt physics-simulaties om agenten in gecontroleerde 3D-omgevingen te trainen.  

Ons project onderscheidt zich door het gebruik van een **eenvoudig CSV-gebaseerd leersysteem** in plaats van complexe neural networks, waardoor het sneller en transparanter kan leren.

---

### Dataset en verkrijging

De dataset bestaat uit **door de AI verzamelde CSV-bestanden** met x-y-paddleposities en bijbehorende slagparameters (strength, angle, spin). Deze worden automatisch tijdens simulaties gegenereerd. Er is geen externe dataset nodig; alle data wordt **real-time geproduceerd en uitgebreid** naarmate meer rally’s worden gespeeld.

Het systeem kijkt bij elke slag of de bal een punt scoort of niet:  
- **Ballen die van de tafel vliegen of in het net belanden** → waardes worden **niet opgeslagen**  
- **Ballen die op de andere kant van de tafel landen** → waardes worden **opgeslagen als bruikbare ervaring**  
- **Ballen die op de andere kant landen én niet worden teruggeslagen** → waardes worden opgeslagen als **extra goede uitkomst**, zodat dit traject vaker kan worden herhaald  

Om variatie in het leerproces te behouden, worden de x-y-punten **afgerond op één decimaal**, waardoor er kwantisatieruis ontstaat. Bovendien worden succesvolle trajecten **verspreid naar naburige posities** met afnemend gewicht (directe buren ~66%, iets verder ~44%, verder gelegen ~22%), zodat kennis sneller wordt gegeneraliseerd.

---

### AI-algoritmen en structuur

We gebruiken een **tabular learning-benadering**: elke x-y-coördinaat correspondeert met een set parameters. Het algoritme kiest parameters op basis van:

- **Exploitatie:** hergebruiken van bekende succesvolle waarden  
- **Exploratie:** willekeurige variatie of nieuwe combinaties  

**Verhoudingen van exploitatie/exploratie:**
- Start: 66% exploitatie / 34% exploratie  
- Wanneer een bal die over dat traject wordt geslagen niet wordt teruggeslagen, wordt de verhouding voor dat punt-specifiek aangepast naar circa 88% exploitatie / 12% exploratie  

Hierdoor leert de AI trajecten die moeilijk terug te slaan zijn vaker te herhalen, vergelijkbaar met het uitbuiten van zwakke plekken van een tegenstander.

Spin wordt **pas toegepast wanneer de bal de tafel raakt**, waardoor trajecten na een bounce extra worden gedraaid. Kleine variaties in strength en angle worden toegevoegd als de bal niet perfect in het midden van de paddle wordt geraakt, zodat het spel realistischer en minder deterministisch blijft.

---

### Tools en libraries

- **PyBullet:** simulatie en physics  
- **Python 3.11:** programmeertaal  
- **Pandas / CSV:** opslag en manipulatie van dataset  
- **NumPy:** berekeningen en interpolaties  
- **Matplotlib:** visualisatie van leercurves

---

### Target en inferentie

De AI draait op **CPU** en inferentie gebeurt **in real-time tijdens de simulatie**. Elke slag wordt berekend en uitgevoerd op basis van de huidige paddlepositie en CSV-data.

---

### Hardware en software

- **Hardware:** standaard laptop of desktop met CPU, 8–16 GB RAM, optioneel GPU voor grafische versnelling  
- **Software:** Windows 10 of Linux, Python 3.11, relevante libraries zoals hierboven

---

### Deployment

De software kan worden gedeployed als:

- Python-script met een requirements.txt of pip-installatie  
- Eventueel gecompileerd als **.exe** via PyInstaller voor standalone gebruik

---

### Samengevat

Door deze combinatie van **simulatie, adaptief leren, data-opslag, exploitatie/exploratie-strategieën, kwantisatie, burenupdate en feedback-gebaseerde CSV-opslag** ontstaat een systeem dat zelfstandig kan leren tafeltennissen in een 3D-omgeving. Het ontwerp maakt het mogelijk om continu data te verzamelen, patronen te herkennen en slagen steeds consistenter en doelgerichter uit te voeren.




##  Resultaat

###  Overzicht van de onderdelen
- **Simulatieomgeving:** Tafel, net, bal en paddles in PyBullet  
- **Slagberekening:** Kracht, hoek, spin, ruis voor variatie  
- **Leersysteem:** CSV-bestanden, exploitatie/exploratie  
- **Evaluatie van prestaties:** Leercurves, rally-observaties, gedragsanalyses  

###  Gedetailleerde resultaten
1. **Simulatie en fysica:**  
   De bal volgt een paraboolfunctie. Trajecten zijn voorspelbaar en consistent, waardoor AI effectief kan leren.  

2. **Prestaties AI:**  
   - Start met willekeurige slagen → veel ballen missen of raken net.  
   - Na ~500 rally’s → 60–70% succesvolle slagen.  
   - Na ~1000 rally’s → 85–90% succesratio; AI gebruikt voornamelijk exploitatie.  

3. **Gedragspatronen:**  
   - AI kiest voorkeurzones waar eerder succes was  
   - Minder extreme hoeken en krachten  
   - Meer consistente rally’s  
   - Lichte variatie zorgt voor realistische dynamiek  

4. **Leercurves:**  
   - Grafieken tonen stijgende lijn van succesratio over rally’s  
   - Fouten nemen af naarmate CSV-bestanden groeien  
   - Exploratie introduceert nieuwe succesvolle trajecten  

> **Figuur 1:** Voorbeeld leercurve over 300 rally’s  
>  <img src="pictures\Schermafbeelding 2025-12-14 180217.png" width="300">  

De resultaten laten zien dat het systeem effectief patronen herkent, leert van ervaring en zich continu verbetert.

##  Uitbreiding 

1. **Geavanceerdere algoritmes**  
   - In de toekomst kan een reinforcement learning-algoritme met Q-learning of policy gradient worden geïmplementeerd.  
   - Dit zou de AI in staat stellen tactische beslissingen te nemen en anticiperen op tegenstanders.

2. **Strategische feedback**  
   - AI kan voorspellingen van tegenstanders gebruiken om de bal op moeilijk bereikbare plekken te slaan.  
   - Dit vereist integratie van extra sensorsimulatie of analyse van eerdere rally’s.

3. **Fysieke robotimplementatie**  
   - Virtueel geleerde slagen kunnen vertaald worden naar een fysieke robotarm.  
   - Dit zou een koppeling vereisen tussen virtuele coördinaten en echte motorbewegingen.

4. **Verbeterde dataset en visualisatie**  
   - Meer variatie in startposities en tegenstanders om het leerproces robuuster te maken.  
   - Grafische dashboards voor realtime monitoring van prestaties.

##  Conclusie

<img src="pictures\Schermafbeelding 2025-12-14 160124.png" width="300">

Het project toont aan dat twee virtuele tafeltennisspelers met een relatief eenvoudig leersysteem hun spel significant kunnen verbeteren. Door parameters en resultaten bij te houden, ontstaat een ervaringsdatabase die toekomstige beslissingen optimaliseert.  

**Doelstellingen vs resultaten:**  
- Het hoofddoel is gehaald: AI leert zelfstandig en adaptief.  
- Subdoelstellingen (simulatie, leersysteem, exploitatie/exploratie, evaluatie) zijn volledig bereikt.  
- Resultaten bewijzen dat het systeem patronen herkent, efficiënter speelt en minder afhankelijk is van toeval.  

**Persoonlijke reflecties:**  
- *Student 1:* Ik heb geleerd hoe feedback en data-opslag een AI-systeem kunnen verbeteren en hoe simulaties essentieel zijn voor veilige en snelle experimenten.  
- *Student 2:* Dit project gaf inzicht in de praktische toepassing van eenvoudige algoritmes en hoe kleine aanpassingen in strategie grote invloed kunnen hebben op prestaties.


##  Bibliografie

- Sutton, R.S., & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.  
- Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.  
- PyBullet Documentation. (n.d.). Retrieved from [https://pybullet.org](https://pybullet.org)  
- Pandas Documentation. (n.d.). Retrieved from [https://pandas.pydata.org](https://pandas.pydata.org)  
- Numpy Documentation. (n.d.). Retrieved from [https://numpy.org](https://numpy.org)




