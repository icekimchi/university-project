#include <stdio.h>
#define MAX_SIZE 10

int fromTop;
int tempTop;
int toTop;

int fromPole[MAX_SIZE];
int tempPole[MAX_SIZE];
int toPole[MAX_SIZE];

int move_count = 0; // ������ �ű� Ƚ���� ��� ���� ���� �Դϴ�.

int getDiskNum();

void initializeDisk(int diskNum);
void moveHanoi(int disks, char from, char temp, char to);
void moveDisk(char from, char to);
void drawAllDisks();

int main(void)
{
    int diskNum = getDiskNum(); //�Լ��� ���� ����ڷκ��� ��ũ ���� �Է� �޴´�.

    while (diskNum > 0)
    {
        initializeDisk(diskNum); //�Լ��� ���� ������ ���ҿ� �ش��ϴ� �迭���� �ʱ�ȭ�Ѵ�.(�迭�� ��������)
        drawAllDisks(); //�Լ��� ���� �ʱ�ȭ�� ������ ����� �׸���.
        moveHanoi(diskNum, 'A', 'B', 'C'); // ����Լ��� ���� ��ũ���� �ű��.

        move_count = 0;

        diskNum = getDiskNum(); //�Լ��� �ٽ� ����ڷκ��� ��ũ ���� �Է� �޾� ���α׷��� �ݺ� �����Ѵ�.
    }
    return 0;
}

int getDiskNum()   //����ڷκ��� ��ũ ������ �Է� �޴´�. 1�̻� MAX_SIZE����
{
    int loop = 1;
    int dish = 0;

    while (loop == 1)
    {
        printf("�ϳ��� ���� ���� �Է��Ͻÿ�. �� : [1<= X <=10]\n��, -1�� �Է��ϸ� ���α׷��� ���� �˴ϴ�. : \n");

        fflush(stdin);
        scanf(" %d", &dish);

        if ((dish >= 1) && (dish <= 10))
        {
            printf("%d���� �������� ���α׷��� ���� �մϴ�.\n", dish);
            loop = 0;

        }

        else if (dish == -1)
        {
            printf("���α׷��� ���� ��ŵ�ϴ�.\n");
            loop = 0;
        }

        else // ������ ���� �Է� �ϸ� �ݺ����� �ٽ� ������ �մϴ�.
        {
            printf("������ �ùٸ��� �ʽ��ϴ�. �ٽ� �Է��Ͻÿ�.\n");
            loop = 1;
        }
    }

    return dish;
}

void initializeDisk(int diskNum)
{
    /*����ڰ� �ʱ� �Է��� ������ ���� ���� ������ ������ �ʱ�ȭ �ϴ� �Լ�
    ù��° ����: ����ڰ� �Է��� ��ũ ��*/

    int i = 0;

    fromTop = diskNum; // ������ ����
    tempTop = 0;
    toTop = 0;

    while (i <= MAX_SIZE) // �ʱ�ȭ
    {
        fromPole[i] = 0;
        tempPole[i] = 0;
        toPole[i] = 0;
        ++i;
    }

    i = 0;
    while (i < fromTop) // ������ ������ �ش�Ǵ� ���ڸ� �����մϴ�.
    {
        fromPole[i] = (diskNum - i); //�ε��� ���� +1 �Ǹ� �迭�� ����Ǵ� ���� -1 �˴ϴ�.
        ++i;
    }

}

void moveHanoi(int disks, char from, char temp, char to)
{
    /*����� ���� �ذ��Ģ�� �°� ����Լ��� �����Ѵ�
    ù���� ����: �ű���� ������ ��
    �ι�° ����: ��� ���ҿ� �ش� �ϴ� ����
    ����° ����: �߰��̿� ������ �̸�
    �׹��� ����: ���� ������ �̸�

    1����Ģ : n-1���� ������ �߰����ҿ� �̵�
    2����Ģ : ��� ������ ���� ������ �Ϻ��� ���� ū ������ �������ҿ� �̵�
    3����Ģ : �߰����ҿ� �ִ� n-1���� ������ ������������ �̵�*/


    if (disks == 1)
    {
        ++move_count;
        printf("%5d: ���� %c���� ���� %c�� ���� %d�� �̵�\n", move_count, from, to, 1);
        moveDisk(from, to);
    }
    else
    {
        moveHanoi(disks - 1, from, to, temp);    // 1����Ģ ����
        ++move_count;
        printf("%5d: ���� %c���� ���� %c�� ���� %d�� �̵�\n", move_count, from, to, disks);  // 2����Ģ ����
        moveDisk(from, to);
        moveHanoi(disks - 1, temp, from, to);  // 3����Ģ ����
    }
}