#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/param.h>
#include <pthread.h>
#include <iostream>

#define TYPE int

using namespace std;

const int POCET_VLAKIEN = 3;   // pocet vlakien
const int POCET_CISEL = 30;   // velkost pola

struct structThread{		//struktura parametrov, ktore idem vlozit do fukcie vlakna na spracovanie
    int id;                 // id vlakna
    int from;				//usek pola od
    int to;					//usek pola do
    TYPE *data;             //cele pole na spracovanie
};

int timeval_to_ms( timeval *before, timeval *after );
void fillArrayRandom(TYPE *cisla, int velkostPole);   // naplni pole nahodnymi hodnotami
void vypisPole(TYPE *cisla, int velkostPole);   // vypise vsetky prvky pola
void bubbleSort(TYPE *cisla, int zaciatok, int koniec,bool smer);   // zotriedi pole, true zostupne, false vzostupne
void *vlaknoTriedenia( void *void_arg );   // funkia pre vlakno na zotriedenie urcitej casti pola
void *vlaknoPlnenia(void *void_arg);
void mergeArrays(TYPE *vystupnePole, int dlzka, TYPE *lavaPolovica, TYPE *pravaPolovica, int lavyIndex, int lavyKoniec, int pravyIndex, int pravyKoniec);// spojenie dvoch zotriedenych poli


int main(){
	pthread_t vlaknoId[POCET_VLAKIEN];		//vytvorim pole vlaken
	pthread_t naplnPole;					//vytvorim vlakno na plneni pole;
    structThread vlaknoStruct[POCET_VLAKIEN];	//vytvorim pole struktur parametrov pre vlakna
    TYPE velkePole[POCET_CISEL];				//zaciatocne pole, ktore neskor naplnime
    TYPE vyslednePole[POCET_CISEL];				//vysledne pole rovnakej velikosti
    timeval casPred,casPo;						//netreba, mozes vymazat

	//naplenie pola cislami
	pthread_create( &naplnPole, NULL, vlaknoPlnenia, &velkePole);    //vytvorime vlakno s nazvom naplnpole, NULL, spracovana funkcia vlaknoplnenia, data na spracovanie cize velke pole do ktoreho vkladame cisla
	pthread_join(naplnPole, NULL );    //synchronizacia vlakna; cakanie na skoncenie vlakna

	//predame strukture potrebne parametre
	for (int i = 0; i < POCET_VLAKIEN; i++){
        vlaknoStruct[i].id = i;      //unikatne id vlakna
		vlaknoStruct[i].data = velkePole;   //priradime pole do vlakna
		vlaknoStruct[i].from = i * (POCET_CISEL/POCET_VLAKIEN); //na zaklade velkosti pola a pocte vlakien vypocitam zaciatocny index casti pola s
		vlaknoStruct[i].to = vlaknoStruct[i].from + (POCET_CISEL/POCET_VLAKIEN);  //vypocitanie konca casti pola pre triedenie
	}
	
	// vytvorenie jednotlivych vlakien
	gettimeofday( &casPred, NULL );      //zmeranie casu na zaciatku
	for (int i = 0; i < POCET_VLAKIEN; i++){
		pthread_create( &vlaknoId[i], NULL, vlaknoTriedenia, &vlaknoStruct[i] );	//vlaknu hodime jeho id,NULL , fuknciu ktora sa v nom bude spracovavat, strukturu pre vlakno
    }
	// synchronizacia jednotlivych vlakien
	for (int i = 0; i < POCET_VLAKIEN; i++){
		pthread_join( vlaknoId[i], NULL );			//cakanie na skoncenie vsetkych vlaken
    }
	
    cout<<"Pole pred:"<<endl;
    vypisPole(velkePole, POCET_CISEL);
    int counter = 0;
    while(true){
        //TYPE poleMin[POCET_VLAKIEN];
        TYPE minPrvok;
        int indexMin;			//index casti, v ktorej sa nachadza momentalne najmensi prvok
        int pocetPrehladanych = 0;
        //bool esteHladaj = false;
        //ziskanie prveho minima
        for(int i = 0; i < POCET_VLAKIEN; i++){
            if(vlaknoStruct[i].from < vlaknoStruct[i].to){
                minPrvok = vlaknoStruct[i].data[vlaknoStruct[i].from];
                indexMin = i;
                break;
            }
        }
        //ziskanie celkoveho minima
        for(int i = 0; i < POCET_VLAKIEN; i++){
            if(vlaknoStruct[i].from < vlaknoStruct[i].to){
                if(vlaknoStruct[i].data[vlaknoStruct[i].from] < minPrvok){
                    minPrvok = vlaknoStruct[i].data[vlaknoStruct[i].from];
                    indexMin = i;
                    //esteHladaj = true;
                }
                pocetPrehladanych++;
            }
        }
        //naplnenie noveho pola
        vlaknoStruct[indexMin].from +=1;
        vyslednePole[counter] = minPrvok;
        counter++;
        if(pocetPrehladanych == 0){
            break;
        }

    }
	gettimeofday( &casPo, NULL );
	cout<<"The sort time: "<< timeval_to_ms( &casPred, &casPo )<<"[ms]"<<endl;

    cout<<"Pole pred:"<<endl;
	vypisPole(vyslednePole, POCET_CISEL);
	return 0;
}

//naplennie pola random cislami
void fillArrayRandom(TYPE *cisla, int velikostPole){
    srand((int)time(NULL));
	for (int i = 0; i < velikostPole; i++){
		cisla[i] = (rand() % 1000 +1 );
		//cout<<cisla[i]<<endl;
	}
}

//klasicky bubblesort
void bubbleSort(TYPE *cisla, int zacatek, int konec,bool smer){
    if(smer){
        for(int i = zacatek; i < konec; i++){
            for(int j = zacatek; j < konec - 1; j++){
                if(cisla[j] > cisla[j+1]){
                    TYPE pom = cisla[j+1];
                    cisla[j+1] = cisla[j];
                    cisla[j] = pom;
                }
            }
        }
	}
	if(!smer){
        for(int i = zacatek; i < konec; i++){
            for(int j = zacatek; j < konec - 1; j++){
                if(cisla[j] > cisla[j+1]){
                    TYPE pom = cisla[j+1];
                    cisla[j+1] = cisla[j];
                    cisla[j] = pom;
                }
            }
        }
	}
}

void *vlaknoPlnenia(void *void_arg){
    TYPE *ptr_data = ( TYPE * ) void_arg;		
    fillArrayRandom(ptr_data,POCET_CISEL);
}

void *vlaknoTriedenia( void *void_arg ){
	structThread *ptr_data = ( structThread * ) void_arg;
	cout<<ptr_data->id<<" "<<ptr_data->from<<" "<<ptr_data->to<<endl;
    bubbleSort(ptr_data->data, ptr_data->from, ptr_data->to,true);
}

void vypisPole(TYPE *cisla, int velikostPole){
	cout<<"Vypis pola: "<<endl;
	for (int i = 0; i < velikostPole; i++){
		cout<<" "<<cisla[i];
	}
	cout<<endl;
	cout<<"Konec vypisu polea"<<endl;
}

void mergeArrays(TYPE *vystupniPole, int delka, TYPE *levaPulka, TYPE *pravaPulka,int levyIndex, int levyKonec, int pravyIndex, int pravyKonec){
    int i = 0;
	for (i = 0; i < delka; i++){
		if (levyIndex >= levyKonec || pravyIndex >= pravyKonec){
			break;
        }
		if (levaPulka[levyIndex] < pravaPulka[pravyIndex]){
			vystupniPole[i] = levaPulka[levyIndex++];
        }
		else{
			vystupniPole[i] = pravaPulka[pravyIndex++];
        }
	}
	while (levyIndex < levyKonec){
		vystupniPole[i++] = levaPulka[levyIndex++];
    }
	while (pravyIndex < pravyKonec){
		vystupniPole[i++] = pravaPulka[pravyIndex++];
    }
}

int timeval_to_ms(timeval *before, timeval *after){
    timeval res;
    timersub( after, before, &res );
    return 1000 * res.tv_sec + res.tv_usec / 1000;
}
