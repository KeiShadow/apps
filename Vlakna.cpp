<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

using namespace std;

#define TYP int

// funkce rand() je pouze 16 bitova
// rand32 vraci 32 bitove nahodne cislo
int rand32()
{
return rand() * RAND_MAX + rand();
}

// funkce pro nalezeni nejvetsiho cisla v useku pole
// od leveho (vcetne) az k pravemu prvku
TYP hledej_max(int levy, int pravy, TYP *pole)
{
TYP nejvetsi = pole[levy];
for (int i = levy + 1; i < pravy; i++)
if (nejvetsi < pole[i])
nejvetsi = pole[i];
return nejvetsi;
}

// Pouziti globalnich promennych zjednodusuje ukazku pouziti vlaken.
// Neni ovsem programatorsky spravne!
// Globalni promenne: DELKA - pocet prvku na ktere odkazuje POLE
int DELKA;
TYP *POLE;
TYP MAX1 = 0, MAX2 = 0;

// vlakno A bude hledat maximalni prvek pole mezi prvkem 0 a DELKA/2
// vysledek ulozi do globalni promenne MAX1
DWORD WINAPI vlaknoA(LPVOID)
{
printf("Startuje vlakno A\n");
MAX1 = hledej_max(0, DELKA / 2, POLE);
return 0;
}

// vlakno B bude hledat maximalni prvek pole mezi prvkem DELKA/2 a DELKA
// vysledek ulozi do globalni promenne MAX2
DWORD WINAPI vlaknoB(LPVOID)
{
printf("Startuje vlakno B\n");
MAX2 = hledej_max(DELKA / 2, DELKA, POLE);
return 0;
}

// vypocet casoveho intervalu v milisekundach
int kolik_ms(LPFILETIME pred, LPFILETIME po)
{
hyper pred64b = pred->dwHighDateTime;
pred64b = (pred64b << 32) | pred->dwLowDateTime;
hyper po64b = po->dwHighDateTime;
po64b = (po64b << 32) | po->dwLowDateTime;
// konverze 100ns -> 1ms
return (int)((po64b - pred64b) / 10000);
}



const int lenght = 10;
int arr[lenght];
int arr1[lenght / 2];
int arr2[lenght / 2];


int printArray(int *arr, int lenght)
{
for (int i = 0; i < lenght; i++)
{
cout << "[" << i << "]: {" << arr[i] << "} " << endl;
}
return 0;
}


void quickSort(int a[], int first, int last);
int pivot(int a[], int first, int last);
void swap(int& a, int& b);
void swapNoTemp(int& a, int& b);
void print(int array[], const int& N);
FILETIME time_start, time_end; // Deklarace promennych pro ziskani casu
int getTime(LPFILETIME start, LPFILETIME end);

DWORD WINAPI threadA(LPVOID);
DWORD WINAPI threadB(LPVOID);

void splitArray(int list[], int list1[], int list2[], int size)
{
int secondhalf = size - size / 2;
for (int i = 0; i < size; i++)
{
if (i < (size / 2))
{
list1[i] = list[i];
}
else
{
list2[i - secondhalf] = list[i];
}
}
}

int main()
{
for (int i = 0; i < (lenght / 2); i++)
{
arr1[i] = rand() % rand32() + 1;
arr2[i] = rand() % rand32() + 1;
}
for (int i = 0; i < (lenght); i++)
{
arr[i] = rand() % rand32() + 1;

}
printArray(arr1, lenght/2);
cout << "" << endl;
printArray(arr2, lenght/2);
cout << "" << endl;
splitArray(arr1, arr2, arr, lenght);
quickSort(arr, 0, lenght - 1);


HANDLE th1, th2;
GetSystemTimeAsFileTime(&time_start);
th1 = CreateThread(0, 0, threadA, 0, 0, 0); // Vytvoreni 1. vlakna
th2 = CreateThread(0, 0, threadB, 0, 0, 0); // Vytvoreni 2. vlakna

WaitForSingleObject(th1, INFINITE); //Cekam na konec 1. vlakna
WaitForSingleObject(th2, INFINITE); //Cekam na konec 2. vlakna

GetSystemTimeAsFileTime(&time_end);



printArray(arr, lenght);
return 0;
}


void quickSort(int a[], int first, int last)
{
int pivotElement;

if (first < last)
{
pivotElement = pivot(a, first, last);
quickSort(a, first, pivotElement - 1);
quickSort(a, pivotElement + 1, last);
}
}


int pivot(int a[], int first, int last)
{
int p = first;
int pivotElement = a[first];

for (int i = first + 1; i <= last; i++)
{
/* If you want to sort the list in the other order, change "<=" to ">" */
if (a[i] <= pivotElement)
{
p++;
swap(a[i], a[p]);
}
}

swap(a[p], a[first]);

return p;
}



void swap(int& a, int& b)
{
int temp = a;
a = b;
b = temp;
}

void swapNoTemp(int& a, int& b)
{
a -= b;
b += a;// b gets the original value of a
a = (b - a);// a gets the original value of b
}

int getTime(LPFILETIME start, LPFILETIME end)
{
hyper start64b = start->dwHighDateTime;
start64b = (start64b << 32) | start->dwLowDateTime;
hyper end64b = end->dwHighDateTime;
end64b = (end64b << 32) | end->dwLowDateTime;
// konverze 100ns -> 1ms
return (int)((end64b - start64b) / 10000);
}



DWORD WINAPI threadA(LPVOID)
{
quickSort(arr1, 0, (lenght - 1) / 2);
return 0;
}
DWORD WINAPI threadB(LPVOID)
{
quickSort(arr2, 0, (lenght - 1) / 2);
return 0;
}
