from intelligent_placer_src import InterlligentPlacer


def main():
    ip = InterlligentPlacer()
    path = '.\..\..\images\\examples\\all\\1.jpg'
    print(f'Is fit: {ip.check_image(path)}')

if __name__ == '__main__':
    main()
