#include <stdio.h>
#include <time.h>

#define MAX_SIZE 10

int fromTop;
int ViaTop;
int toTop;

int fromPole[MAX_SIZE];
int ViaPole[MAX_SIZE];
int toPole[MAX_SIZE];
int move_count = 0; // 원반을 옮긴 횟수를 출력 해줄 변수 입니다.
int get_Disk_Num();

void initialize_Disk(int diskNum);
void move_Hanoi(int disks, char from, char Via, char to);
void move_Disk(char from, char to);
void draw_All_Disks();
void pyramid(int number);

void delay(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}

int main(void)
{
    int diskNum = get_Disk_Num(); //함수를 통해 사용자로부터 디스크 수를 입력 받는다.

    while (diskNum > 0)
    {
        initialize_Disk(diskNum); //함수를 통해 각각의 말뚝에 해당하는 배열들을 초기화한다.(배열은 전역변수)
        draw_All_Disks(); //함수를 통해 초기화된 말뚝의 모습을 그린다.
        move_Hanoi(diskNum, 'A', 'B', 'C'); // 재귀함수를 통해 디스크들을 옮긴다.

        move_count = 0;
        diskNum = get_Disk_Num(); //함수로 다시 사용자로부터 디스크 수를 입력 받아 프로그램을 반복 수행한다.
    }
    return 0;
}

int get_Disk_Num()
{ //사용자로부터 디스크 개수를 입력 받는다. 1이상 MAX_SIZE이하
    int loop = 1;
    int dish = 0;

    while (loop == 1)
    {
        printf("\n하노이 원반 수를 입력하시오(3<= X <=5) \n\n(단, -1을 입력하면 프로그램은 종료 됩니다): ");

        fflush(stdin);
        scanf_s("%d", &dish);

        if ((dish >= 1) && (dish <= 5))
        {
            printf("\n%d개의 원반으로 하노이 탑 프로그램을 실행\n", dish);
            loop = 0;
        }

        else if (dish == -1)
        {
            printf("프로그램을 종료합니다.\n");
            loop = 0;
        }

        else // 비정상 값을 입력 하면 반복문을 다시 돌도록 합니다.
        {
            printf("범위 외의 값입니다. 다시 입력하시오.\n");
            loop = 1;
        }
    }
    return dish;
}

void initialize_Disk(int diskNum)
{
    /*사용자가 초기 입력한 원반의 수에 따라 각각의 말뚝을 초기화 하는 함수
    첫번째 인자: 사용자가 입력한 디스크 수*/
    int i = 0;

    fromTop = diskNum; // 원반의 개수
    ViaTop = 0;
    toTop = 0;

    while (i <= MAX_SIZE) // 초기화
    {
        fromPole[i] = 0;
        ViaPole[i] = 0;
        toPole[i] = 0;
        ++i;
    }

    i = 0;
    while (i < fromTop) // 원반의 개수에 해당되는 숫자를 저장합니다.
    {
        fromPole[i] = (diskNum - i); //인덱스 값이 +1 되면 배열에 저장되는 수는 -1 됩니다.
        ++i;
    }
}

void move_Hanoi(int disks, char from, char Via, char to)
{
    if (disks == 1)
    {
        ++move_count;
        printf("원반 %d(을)를 말뚝 %c에서 말뚝 %c(으)로 이동 (횟수: %2d)\n", 1, from, to, move_count);
        move_Disk(from, to);
    }
    else
    {
        move_Hanoi(disks - 1, from, to, Via);    // 1번규칙 적용
        ++move_count;
        printf("원반 %d(을)를 말뚝 %c에서 말뚝 %c(으)로 이동 (횟수: %2d)\n", disks, from, to, move_count);  // 2번규칙 적용
        move_Disk(from, to);
        move_Hanoi(disks - 1, Via, from, to);  // 3번규칙 적용
    } 
}

void move_Disk(char from, char to)
{
    int* FROM, * TO; //포인터를 지정합니다. 매개변수 인자의 값에 따라 가르키는 값이 다르게 됩니다.
    int From_index, To_index; // 배열에 값을 옮길 때 쓰일 임시 변수입니다.

    if (from == 'A')
    {
        FROM = fromPole;
        From_index = fromTop--;
    }
    else if (from == 'B')
    {
        FROM = ViaPole;
        From_index = ViaTop--;
    }
    else //(from == 'C')
    {
        FROM = toPole;
        From_index = toTop--;
    }

    if (to == 'A')
    {
        TO = fromPole;
        To_index = fromTop++;
    }
    else if (to == 'B')
    {
        TO = ViaPole;
        To_index = ViaTop++;
    }
    else
    {
        TO = toPole;
        To_index = toTop++;
    }
    TO[To_index] = FROM[From_index - 1]; //배열에 값을 쓴 후에
    FROM[From_index - 1] = 0; // 이전에 있던 배열에서는 0으로 삭제 합니다.
    draw_All_Disks();
}

void draw_All_Disks()
{
    int j = 1;
    int height;
    height = ((fromTop >= ViaTop) ? fromTop : ViaTop);
    height = ((toTop >= height) ? toTop : height);
    // 최대 값을 결정합니다. 이 값은 원반을 출력시 높낮이를 결정합니다.
    delay(750);

    while (height >= 0) //행의 역할을 한다. (tower_height -> 0  인덱스 순으로 부터 출력한다.)
    {
        pyramid(fromPole[height]);
        printf("\t");
        pyramid(ViaPole[height]);  
        printf("\t");
        pyramid(toPole[height]);
        printf("\n");

        height--;
    }
    printf("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\tㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\tㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n         ⓐ         \t         ⓑ         \t         ⓒ         \n\n");
}// 일부러 인덱스 하나를 더 출력 하게 해서 printf("\n"); 을 또 사용하지 않도록 했다.

void pyramid(int number) {
    switch (number) {
    case 0:
        printf("                    ");
        break;
    case 1:
        printf("         ♡         ");
        break;
    case 2:
        printf("        ♡♡        ");
        break;
    case 3:
        printf("       ♡♡♡       ");
        break;
    case 4:
        printf("      ♡♡♡♡      ");
        break;
    case 5:
        printf("     ♡♡♡♡♡     ");
        break;
    default:
        break;
    }
}