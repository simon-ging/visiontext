import pygame
from pygame_gui import UIManager
from pygame_gui.elements import UIScrollingContainer, UITextBox


def main():
    pygame.init()
    pygame.display.set_caption("Quick Start")
    window_surface = pygame.display.set_mode((800, 600))
    manager = UIManager((800, 600))
    background = pygame.Surface((800, 600))
    background.fill(manager.ui_theme.get_colour("dark_bg"))
    scrolling_container = UIScrollingContainer(pygame.Rect(0, 0, 200, 200))
    scrolling_container.set_scrollable_area_dimensions((400, 400))
    text_box = UITextBox(
        html_text="examples.holmes_text_test " * 50, relative_rect=pygame.Rect(300, 0, 200, 200)
    )
    clock = pygame.time.Clock()
    is_running = True
    while is_running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            manager.process_events(event)
        manager.update(time_delta)
        window_surface.blit(background, (0, 0))
        manager.draw_ui(window_surface)
        pygame.display.update()


if __name__ == "__main__":
    main()
