// base scripts
var tablinks = document.getElementsByClassName("tab-links");
var tabcontents = document.getElementsByClassName("tab-contents");
function opentab(tabname) {
    for (tablink of tablinks) {
        tablink.classList.remove("active-link");
    }
    for (tabcontent of tabcontents) {
        tabcontent.classList.remove("active-tab");
    }
    event.currentTarget.classList.add("active-link");
    document.getElementById(tabname).classList.add("active-tab")
}
























// <!-- portfolio script -->
function toggle_portfolio(index) {
    const blur_container = document.getElementById('portfolioblur');
    blur_container.classList.toggle('portfolioactive');
    const popupId = `popup-${index}`;
    const popup_box = document.getElementById(popupId);
    popup_box.classList.remove('animate');
    popup_box.classList.remove('close');
    if (popup_box.style.display === 'block') {
        popup_box.classList.add('close');
        setTimeout(() => {
            popup_box.style.display = 'none';
            popup_box.classList.remove('visible');
        }, 500); // Match the animation duration
        document.body.classList.remove('no-scroll');
        document.body.style.overflowY = 'auto';
    } else {
        popup_box.classList.add('animate');
        popup_box.classList.add('visible');
        popup_box.style.display = 'block';
        document.body.classList.add('no-scroll');
        document.body.style.overflowY = 'hidden';
    }
    if (popup_box.style.display === 'block') {
        currentIndex = 0; // Reset the index when opening a new popup
        showImage(currentIndex); // Show the first image
    }
    // Ensure event object is passed
    if (event) event.preventDefault();
}
    // Add a click event listener to the "read more" button
    document.querySelectorAll('.read-more').forEach(button => {
        // Add the animate class to the popup element
        button.addEventListener('click', () => {
            const popupId = button.parentNode.parentNode.dataset.portfoliopopup;
            const popup = document.getElementById(popupId);
            popup.classList.remove('close');
            popup.classList.add('animate');
            popup.style.display = 'block';
            document.body.classList.add('no-scroll');
            document.body.style.overflowY = 'hidden';
        });
    });
// Event listener to close the popup when clicking outside
document.addEventListener('click', function (event) {
    const popupBox = document.querySelector('.portfoliopopup.visible');
    if (popupBox && !popupBox.contains(event.target) && !event.target.closest('.grid-item')) {
        toggle_portfolio(popupBox.id.split('-')[1]); // Close the popup if clicking outside
    }
});
// slider
let currentIndex = 0;
const images = document.querySelectorAll('.image-slider img');
console.log(`Total number of images: ${images.length}`);
function showImage(sliderId, index) {
    var slider = document.getElementById(sliderId);
    var images = slider.getElementsByTagName('img');
    for (var i = 0; i < images.length; i++) {
        images[i].style.display = 'none';
    }
    images[index].style.display = 'block';
}
function nextImage(sliderId) {
    var slider = document.getElementById(sliderId);
    var images = slider.getElementsByTagName('img');
    var currentImageIndex = Array.from(images).findIndex(img => img.style.display === 'block');
    var newIndex = (currentImageIndex + 1) % images.length;
    showImage(sliderId, newIndex);
}
function prevImage(sliderId) {
    var slider = document.getElementById(sliderId);
    var images = slider.getElementsByTagName('img');
    var currentImageIndex = Array.from(images).findIndex(img => img.style.display === 'block');
    var newIndex = (currentImageIndex - 1 + images.length) % images.length;
    showImage(sliderId, newIndex);
}
// Initialize the first image as visible in each slider on page load
window.onload = function () {
    var sliders = document.querySelectorAll('.image-slider');
    sliders.forEach(function (slider) {
        var images = slider.getElementsByTagName('img');
        if (images.length > 0) {
            images[0].style.display = 'block';
        }
    });
};
// Event listeners for navigation buttons
document.querySelector('.next').addEventListener('click', nextImage);
document.querySelector('.prev').addEventListener('click', prevImage);



// <!-- Certificates -->
function toggle_certificate(index) {
    const blur = document.getElementById('certificateblur');
    blur.classList.toggle('certificateactive');

    const popupId = `popup-${index}`;
    const popup_box = document.getElementById(popupId);
    
    popup_box.classList.remove('animate');
    popup_box.classList.remove('close');
    
    if (popup_box.style.display === 'block') {
        popup_box.classList.add('close');
        setTimeout(() => {
            popup_box.style.display = 'none';
            popup_box.classList.remove('visible');
        }, 500); // Match the animation duration
        document.body.classList.remove('no-scroll');
        document.body.style.overflowY = 'auto';
    } else {
        popup_box.classList.add('animate');
        popup_box.classList.add('visible');
        popup_box.style.display = 'block';
        document.body.classList.add('no-scroll');
        document.body.style.overflowY = 'hidden';
    }
    // Ensure event object is passed
    // if (event) event.preventDefault();
}
// Event listener to toggle the no-scroll class when the popup is closed
document.addEventListener('click', function (event) {
    if (event.target.classList.contains('close') || event.target.closest('.certificatepopup')) {
        document.body.classList.remove('no-scroll');
        document.body.style.overflowY = 'auto';
    }
});
// Add a click event listener to the "read more" button
document.querySelectorAll('.click-to-view').forEach(button => {
    // Add the animate class to the popup element
    button.addEventListener('click', () => {
        const popupId = button.parentNode.parentNode.dataset.certificatepopup;
        const popup = document.getElementById(popupId);
        popup.classList.remove('close');
        popup.classList.add('animate');
        popup.style.display = 'block';
        document.body.classList.add('no-scroll');
        document.body.style.overflowY = 'hidden';
    });
});
// Event listener to close the popup when clicking outside
document.addEventListener('click', function (event) {
    const popupBox = document.querySelector('.certificatepopup.visible');
    if (popupBox && !popupBox.contains(event.target) && !event.target.closest('.grid-item')) {
        toggle_certificate(popupBox.id.split('-')[1]); // Close the popup if clicking outside
    }
});

