from logic import *

clauses = []

### I1. If the animal has hair then it is a mammal
clauses.append(expr("(Hair(x) ==> Mammal(x))"))

### I2. If the animal gives milk then it is a mammal
clauses.append(expr("(GivesMilk(x) ==> Mammal(x))"))

### I3. If the animal has feathers then it is a bird
clauses.append(expr("(HasFeathers(x) ==> Bird(x))"))

### I4. If the animal flies and it lays eggs then it is a bird
clauses.append(expr("(Flies(x) & LaysEggs(x) ==> Bird(x))"))

### I5. If the animal is a mammal and it eats meat then it is a carnivore
clauses.append(expr("(Mammal(x) & EatsMeat(x) ==> Carnivore(x))"))

### I6. If the animal is a mammal and it has pointed teeth and it has claws and its eyes point forward
###   then it is a carnivore
clauses.append(expr("(Mammal(x) & PointedTeeth(x) & HasClaws(x) & EyesPointForward(x) ==> Carnivore(x))"))

### I7. If the animal is a mammal and it has hoofs then it is an ungulate
clauses.append(expr("(Mammal(x) & HasHoofs(x) ==> Ungulate(x))"))

### 18. If the animal is a mammal and it chews cud then it is an ungulate
clauses.append(expr("(Mammal(x) & ChewsCud(x) ==> Ungulate(x))"))

### I9. If the animal is a carnivore and it has a tawny color and it has dark spots then it is a cheetah
clauses.append(expr("(Carnivore(x) & TawnyColor(x) & DarkSpots(x) ==> Cheetah(x))"))

### I10. If the animal is a carnivore and it has a tawny color and it has black stripes then it is a tiger
clauses.append(expr("(Carnivore(x) & TawnyColor(x) & BlackStripes(x) ==> Tiger(x))"))

### I11. If the animal is an ungulate and it has long legs and it has a long neck and it has a tawny color
###   and it has dark spots then it is a giraffe
clauses.append(expr("(Ungulate(x) & TawnyColor(x) & LongLegs(x) & LongNeck(x) & DarkSpots(x) ==> Giraffe(x))"))

### I12. If the animal is an ungulate and it has a white color and it has black stripes then it is a zebra
clauses.append(expr("(Ungulate(x) & WhiteColor(x) & BlackStripes(x) ==> Zebra(x))"))

### Il3. If the animal is a bird and it does not fly and it has long legs and it has a long neck and it is black
###   and white then it is an ostrich,
clauses.append(expr("(Bird(x) & DoesNotFly(x) & LongLegs(x) & LongNeck(x) & BlackAndWhite(x) ==> Ostrich(x))"))

### Il4. If the animal is a bird and it does not fly and it swims and it is black and white then it is a
###    penguin
clauses.append(expr("(Bird(x) & DoesNotFly(x) & Swims(x) & BlackAndWhite(x) ==> Penguin(x))"))

### Il5. If the animal is a bird and it is a good flyer then it is an albatross.
clauses.append(expr("(Bird(x) & GoodFlyer(x) ==> Albatross(x))"))

animals_kb = FolKB(clauses)

animals_kb.tell(expr("GivesMilk(Animal1)"))
animals_kb.tell(expr("HasHoofs(Animal1)"))
animals_kb.tell(expr("EatsMeat(Animal1)"))
animals_kb.tell(expr("TawnyColor(Animal1)"))
animals_kb.tell(expr("BlackStripes(Animal1)"))

animals_kb.tell(expr("GivesMilk(Animal2)"))
animals_kb.tell(expr("HasHoofs(Animal2)"))
animals_kb.tell(expr("EatsMeat(Animal2)"))
animals_kb.tell(expr("WhiteColor(Animal2)"))
animals_kb.tell(expr("BlackStripes(Animal2)"))

animals_kb.tell(expr("Flies(Animal3)"))
animals_kb.tell(expr("LaysEggs(Animal3)"))
animals_kb.tell(expr("GoodFlyer(Animal3)"))

animals_kb.tell(expr("HasFeathers(Animal4)"))
animals_kb.tell(expr("DoesNotFly(Animal4)"))
animals_kb.tell(expr("Swims(Animal4)"))
animals_kb.tell(expr("BlackAndWhite(Animal4)"))

print("Mammal:", list(fol_bc_ask(animals_kb, expr('Mammal(y)'))))
print("Carnivore:", list(fol_bc_ask(animals_kb, expr('Carnivore(y)'))))
print("Zebra:", list(fol_bc_ask(animals_kb, expr('Zebra(y)'))))
print("Cheetah:", list(fol_fc_ask(animals_kb, expr('Cheetah(y)'))))
print("Tiger:", list(fol_fc_ask(animals_kb, expr('Tiger(y)'))))
print("Giraffe:", list(fol_fc_ask(animals_kb, expr('Giraffe(y)'))))
print("Bird:", list(fol_bc_ask(animals_kb, expr('Bird(y)'))))
print("Ostrich:", list(fol_bc_ask(animals_kb, expr('Ostrich(y)'))))
print("Penguin:", list(fol_bc_ask(animals_kb, expr('Penguin(y)'))))
print("Albatross:", list(fol_bc_ask(animals_kb, expr('Albatross(y)'))))
