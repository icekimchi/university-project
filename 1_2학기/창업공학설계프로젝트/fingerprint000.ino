#include <Adafruit_Fingerprint.h>
#include <Servo.h>
#include <Keypad.h>

//서브모터 설정
Servo servo;
int motor = 13;
int angle = 90;

//피에조부저
int piezo = 4 ;
int numTones = 3;
int tones1[] = {293, 391, 493};
int tones2[] = {493, 493, 493};
double b[] = {0.5, 0.5, 0.5};

//led 설정
int led1 = A0;
int led2 = A1;

//키패드설정
int tru=0;
int count=0;
char PW[4] = {'2','2','2','2'};
const byte ROWS = 4;   
const byte COLS = 4;   
char hexaKeys[ROWS][COLS] = {  
  {'1', '2', '3', 'A'},  
  {'4', '5', '6', 'B'},  
  {'7', '8', '9', 'C'},  
  {'*', '0', '#', 'D'}};  
byte rowPins[ROWS] = {12, 11, 10, 9};   
byte colPins[COLS] = {8, 7, 6, 5};  
Keypad customKeypad = Keypad(makeKeymap(hexaKeys), rowPins, colPins, ROWS, COLS);   
char keyPressed;  
  
#if (defined(__AVR__) || defined(ESP8266)) && !defined(__AVR_ATmega2560__)
SoftwareSerial mySerial(2, 3);

#else
#define mySerial Serial1
#endif
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&mySerial);

void setup()
{
  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);
  servo.attach(motor);
  servo.write(90);
  Serial.begin(9600);
  pinMode(piezo, OUTPUT);
  
  while (!Serial);
  delay(100);

  finger.begin(57600);
  delay(5);
  if (finger.verifyPassword()) {
    Serial.println("Found fingerprint sensor!");
  }
  else {
    Serial.println("Did not find fingerprint sensor :(");
    while (1) { delay(1); }
  }

  finger.getParameters();
}

void loop()                
{
  getFingerprintID();  
  keyPressed = customKeypad.getKey();
  if (keyPressed){
    Serial.println(keyPressed);  
    keypadpress();
  }
  delay(1000);
}

uint8_t getFingerprintID() {
  uint8_t p = finger.getImage();
  switch (p) {
    case FINGERPRINT_OK:
      Serial.println("Image taken");
      break;
    case FINGERPRINT_NOFINGER:
      Serial.println("No finger detected");
      return p;
    case FINGERPRINT_PACKETRECIEVEERR:
      Serial.println("Communication error");
      return p;
    case FINGERPRINT_IMAGEFAIL:
      Serial.println("Imaging error");
      return p;
    default:
      Serial.println("Unknown error");
      return p;
  }

  p = finger.image2Tz();
  switch (p) {
    case FINGERPRINT_OK:
      Serial.println("Image converted");
      break;
    case FINGERPRINT_IMAGEMESS:
      Serial.println("Image too messy");
      errorRing();
      return p;
    case FINGERPRINT_PACKETRECIEVEERR:
      Serial.println("Communication error");
      return p;
    case FINGERPRINT_FEATUREFAIL:
      Serial.println("Could not find fingerprint features");
      errorRing();
      return p;
    case FINGERPRINT_INVALIDIMAGE:
      Serial.println("Could not find fingerprint features");
      errorRing();
      return p;
    default:
      Serial.println("Unknown error");
      errorRing();
      return p;
  }

  p = finger.fingerSearch();
  if (p == FINGERPRINT_OK) {
    Serial.println("Found a print match!");
    correctRing();
    openDoor();
  } else if (p == FINGERPRINT_PACKETRECIEVEERR) {
    Serial.println("Communication error");
    errorRing();
    return p;
  } else if (p == FINGERPRINT_NOTFOUND) {
    Serial.println("Did not find a match");
    errorRing();
    return p;
  } else {
    Serial.println("Unknown error");
    errorRing();
    return p;
  }
  Serial.print("Found ID #"); Serial.print(finger.fingerID);
  Serial.print(" with confidence of "); Serial.println(finger.confidence);

  return finger.fingerID;
}

int getFingerprintIDez() {
  uint8_t p = finger.getImage();
  if (p != FINGERPRINT_OK)  return -1;

  p = finger.image2Tz();
  if (p != FINGERPRINT_OK)  return -1;

  p = finger.fingerFastSearch();
  if (p != FINGERPRINT_OK)  return -1;

  Serial.print("Found ID #"); Serial.print(finger.fingerID);
  Serial.print(" with confidence of "); Serial.println(finger.confidence);
  return finger.fingerID;
}

void correctRing(){
  for(int i=0; i<numTones; i++){
    if (tones1[i]!=0){
      tone(piezo, tones1[i]);
      delay(b[i]*300);
      noTone(piezo);
      delay(b[i]*100);
      }
    else delay(b[i]*400);}
}

void errorRing(){
    for(int i=0; i<numTones; i++){
        if (tones2[i]!=0){
          tone(piezo, tones2[i]);
          delay(b[i]*300);
          noTone(piezo);
          delay(b[i]*100);
        }
        else delay(b[i]*400);}  
}

void openDoor(){
    servo.write(90);
    turnlight();
    for(int i=0; i<91; i++){
      angle = angle + 1;
      servo.write(angle);
      delay(10);
    }
    delay(9000);
    offlight();
    servo.write(90);
}

void keypadpress(){
    Serial.println(keyPressed);  
    if(keyPressed == PW[count]){
      count++;
      tru++;  
    }
    else if(keyPressed != PW[count]){
      count++;
    } 
    if(keyPressed=='#'){
      tru=0;
      count=0;
    }
    if(count==4){
      if (tru==4){
        correctRing();
        openDoor();
        offlight();
      }
      else{
        errorRing();
      }
      tru = 0;
      count = 0;
   } 
}  

void turnlight(){
  digitalWrite(led1, HIGH);
  digitalWrite(led2, HIGH);
}
void offlight(){
  digitalWrite(led1, LOW);
  digitalWrite(led2, LOW);
}
