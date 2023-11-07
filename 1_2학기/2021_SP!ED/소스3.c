#include <stdio.h>
#include <time.h>

#define MAX_SIZE 10

int fromTop;
int ViaTop;
int toTop;

int fromPole[MAX_SIZE];
int ViaPole[MAX_SIZE];
int toPole[MAX_SIZE];
int move_count = 0; // ������ �ű� Ƚ���� ��� ���� ���� �Դϴ�.
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
    int diskNum = get_Disk_Num(); //�Լ��� ���� ����ڷκ��� ��ũ ���� �Է� �޴´�.

    while (diskNum > 0)
    {
        initialize_Disk(diskNum); //�Լ��� ���� ������ ���ҿ� �ش��ϴ� �迭���� �ʱ�ȭ�Ѵ�.(�迭�� ��������)
        draw_All_Disks(); //�Լ��� ���� �ʱ�ȭ�� ������ ����� �׸���.
        move_Hanoi(diskNum, 'A', 'B', 'C'); // ����Լ��� ���� ��ũ���� �ű��.

        move_count = 0;
        diskNum = get_Disk_Num(); //�Լ��� �ٽ� ����ڷκ��� ��ũ ���� �Է� �޾� ���α׷��� �ݺ� �����Ѵ�.
    }
    return 0;
}

int get_Disk_Num()
{ //����ڷκ��� ��ũ ������ �Է� �޴´�. 1�̻� MAX_SIZE����
    int loop = 1;
    int dish = 0;

    while (loop == 1)
    {
        printf("\n�ϳ��� ���� ���� �Է��Ͻÿ�(3<= X <=5) \n\n(��, -1�� �Է��ϸ� ���α׷��� ���� �˴ϴ�): ");

        fflush(stdin);
        scanf_s("%d", &dish);

        if ((dish >= 1) && (dish <= 5))
        {
            printf("\n%d���� �������� �ϳ��� ž ���α׷��� ����\n", dish);
            loop = 0;
        }

        else if (dish == -1)
        {
            printf("���α׷��� �����մϴ�.\n");
            loop = 0;
        }

        else // ������ ���� �Է� �ϸ� �ݺ����� �ٽ� ������ �մϴ�.
        {
            printf("���� ���� ���Դϴ�. �ٽ� �Է��Ͻÿ�.\n");
            loop = 1;
        }
    }
    return dish;
}

void initialize_Disk(int diskNum)
{
    /*����ڰ� �ʱ� �Է��� ������ ���� ���� ������ ������ �ʱ�ȭ �ϴ� �Լ�
    ù��° ����: ����ڰ� �Է��� ��ũ ��*/
    int i = 0;

    fromTop = diskNum; // ������ ����
    ViaTop = 0;
    toTop = 0;

    while (i <= MAX_SIZE) // �ʱ�ȭ
    {
        fromPole[i] = 0;
        ViaPole[i] = 0;
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

void move_Hanoi(int disks, char from, char Via, char to)
{
    if (disks == 1)
    {
        ++move_count;
        printf("���� %d(��)�� ���� %c���� ���� %c(��)�� �̵� (Ƚ��: %2d)\n", 1, from, to, move_count);
        move_Disk(from, to);
    }
    else
    {
        move_Hanoi(disks - 1, from, to, Via);    // 1����Ģ ����
        ++move_count;
        printf("���� %d(��)�� ���� %c���� ���� %c(��)�� �̵� (Ƚ��: %2d)\n", disks, from, to, move_count);  // 2����Ģ ����
        move_Disk(from, to);
        move_Hanoi(disks - 1, Via, from, to);  // 3����Ģ ����
    } 
}

void move_Disk(char from, char to)
{
    int* FROM, * TO; //�����͸� �����մϴ�. �Ű����� ������ ���� ���� ����Ű�� ���� �ٸ��� �˴ϴ�.
    int From_index, To_index; // �迭�� ���� �ű� �� ���� �ӽ� �����Դϴ�.

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
    TO[To_index] = FROM[From_index - 1]; //�迭�� ���� �� �Ŀ�
    FROM[From_index - 1] = 0; // ������ �ִ� �迭������ 0���� ���� �մϴ�.
    draw_All_Disks();
}

void draw_All_Disks()
{
    int j = 1;
    int height;
    height = ((fromTop >= ViaTop) ? fromTop : ViaTop);
    height = ((toTop >= height) ? toTop : height);
    // �ִ� ���� �����մϴ�. �� ���� ������ ��½� �����̸� �����մϴ�.
    delay(750);

    while (height >= 0) //���� ������ �Ѵ�. (tower_height -> 0  �ε��� ������ ���� ����Ѵ�.)
    {
        pyramid(fromPole[height]);
        printf("\t");
        pyramid(ViaPole[height]);  
        printf("\t");
        pyramid(toPole[height]);
        printf("\n");

        height--;
    }
    printf("�ѤѤѤѤѤѤѤѤѤ�\t�ѤѤѤѤѤѤѤѤѤ�\t�ѤѤѤѤѤѤѤѤѤ�\n         ��         \t         ��         \t         ��         \n\n");
}// �Ϻη� �ε��� �ϳ��� �� ��� �ϰ� �ؼ� printf("\n"); �� �� ������� �ʵ��� �ߴ�.

void pyramid(int number) {
    switch (number) {
    case 0:
        printf("                    ");
        break;
    case 1:
        printf("         ��         ");
        break;
    case 2:
        printf("        ����        ");
        break;
    case 3:
        printf("       ������       ");
        break;
    case 4:
        printf("      ��������      ");
        break;
    case 5:
        printf("     ����������     ");
        break;
    default:
        break;
    }
}