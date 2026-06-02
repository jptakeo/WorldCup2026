document.addEventListener("DOMContentLoaded", function () {
    const menuToggle = document.querySelector(".menu-toggle");
    const menu = document.querySelector(".menu");
    const dropdown = document.querySelector(".menu li.dropdown");
    const dropdownLink = document.querySelector(".dropdown-link");

    const isMobile = () => window.matchMedia("(max-width: 1024px)").matches;

    if (menuToggle && menu) {
        menuToggle.addEventListener("click", function () {
            menuToggle.classList.toggle("active");
            menu.classList.toggle("open");

            if (!menu.classList.contains("open") && dropdown && dropdownLink) {
                dropdown.classList.remove("open");
                dropdownLink.setAttribute("aria-expanded", "false");
            }
        });
    }

    if (dropdown && dropdownLink) {
        dropdownLink.addEventListener("click", function (event) {
            event.preventDefault();

            if (!isMobile()) return;

            dropdown.classList.toggle("open");

            const isOpen = dropdown.classList.contains("open");
            dropdownLink.setAttribute("aria-expanded", String(isOpen));
        });
    }

    window.addEventListener("resize", function () {
        if (!isMobile() && dropdown && dropdownLink) {
            dropdown.classList.remove("open");
            dropdownLink.setAttribute("aria-expanded", "false");
        }
    });
});