#include <stdio.h>
#define MAX_SIZE 10

int fromTop;
int tempTop;
int toTop;

int fromPole[MAX_SIZE];
int tempPole[MAX_SIZE];
int toPole[MAX_SIZE];

int move_count = 0; // 원반을 옮긴 횟수를 출력 해줄 변수 입니다.

int getDiskNum();

void initializeDisk(int diskNum);
void moveHanoi(int disks, char from, char temp, char to);
void moveDisk(char from, char to);
void drawAllDisks();

int main(void)
{
    int diskNum = getDiskNum(); //함수를 통해 사용자로부터 디스크 수를 입력 받는다.

    while (diskNum > 0)
    {
        initializeDisk(diskNum); //함수를 통해 각각의 말뚝에 해당하는 배열들을 초기화한다.(배열은 전역변수)
        drawAllDisks(); //함수를 통해 초기화된 말뚝의 모습을 그린다.
        moveHanoi(diskNum, 'A', 'B', 'C'); // 재귀함수를 통해 디스크들을 옮긴다.

        move_count = 0;

        diskNum = getDiskNum(); //함수로 다시 사용자로부터 디스크 수를 입력 받아 프로그램을 반복 수행한다.
    }
    return 0;
}

int getDiskNum()   //사용자로부터 디스크 개수를 입력 받는다. 1이상 MAX_SIZE이하
{
    int loop = 1;
    int dish = 0;

    while (loop == 1)
    {
        printf("하노이 원반 수를 입력하시오. 값 : [1<= X <=10]\n단, -1을 입력하면 프로그램은 종료 됩니다. : \n");

        fflush(stdin);
        scanf(" %d", &dish);

        if ((dish >= 1) && (dish <= 10))
        {
            printf("%d개의 원반으로 프로그램을 실행 합니다.\n", dish);
            loop = 0;

        }

        else if (dish == -1)
        {
            printf("프로그램을 종료 시킵니다.\n");
            loop = 0;
        }

        else // 비정상 값을 입력 하면 반복문을 다시 돌도록 합니다.
        {
            printf("범위가 올바르지 않습니다. 다시 입력하시오.\n");
            loop = 1;
        }
    }

    return dish;
}

void initializeDisk(int diskNum)
{
    /*사용자가 초기 입력한 원반의 수에 따라 각각의 말뚝을 초기화 하는 함수
    첫번째 인자: 사용자가 입력한 디스크 수*/

    int i = 0;

    fromTop = diskNum; // 원반의 개수
    tempTop = 0;
    toTop = 0;

    while (i <= MAX_SIZE) // 초기화
    {
        fromPole[i] = 0;
        tempPole[i] = 0;
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

void moveHanoi(int disks, char from, char temp, char to)
{
    /*재귀적 문제 해결규칙에 맞게 재귀함수를 구현한다
    첫번재 인자: 옮기려는 원반의 수
    두번째 인자: 출발 말뚝에 해당 하는 문자
    세번째 인자: 중간이용 말뚝의 이름
    네번재 인자: 목적 말뚝의 이름

    1번규칙 : n-1개의 원반을 중간말뚝에 이동
    2번규칙 : 출발 말뚝의 가장 마지막 하부의 제일 큰 원반을 목적말뚝에 이동
    3번규칙 : 중간말뚝에 있는 n-1개의 원반을 목적말뚝으로 이동*/


    if (disks == 1)
    {
        ++move_count;
        printf("%5d: 말뚝 %c에서 말뚝 %c로 원반 %d를 이동\n", move_count, from, to, 1);
        moveDisk(from, to);
    }
    else
    {
        moveHanoi(disks - 1, from, to, temp);    // 1번규칙 적용
        ++move_count;
        printf("%5d: 말뚝 %c에서 말뚝 %c로 원반 %d를 이동\n", move_count, from, to, disks);  // 2번규칙 적용
        moveDisk(from, to);
        moveHanoi(disks - 1, temp, from, to);  // 3번규칙 적용
    }
}