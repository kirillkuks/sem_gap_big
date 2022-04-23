from intelligent_placer_src import IntelligentPlacer


def main():
    ip = IntelligentPlacer('.\..\..\\images\\objects\\')

    path = '.\..\..\images\\examples\\all\\1.jpg'
    
    print(f'Is fit: {ip.check_image(path)}')


if __name__ == '__main__':
    main()
