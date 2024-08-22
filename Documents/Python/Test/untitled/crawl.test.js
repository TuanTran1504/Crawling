const {normalizeURL, getURLsFromHTML} = require('./crawl.js')
const {test, expect} = require('@jest/globals')

test('normalizeURL strip protocol', () => {
    const input ='https://blog.boot.dev/path/'
    const actual = normalizeURL(input)
    const expected='blog.boot.dev/path'
    expect(actual).toEqual(expected)
})

test('normalizeURL strip trailing', () => {
    const input ='https://blog.boot.dev/path/'
    const actual = normalizeURL(input)
    const expected='blog.boot.dev/path'
    expect(actual).toEqual(expected)
})
test('getURLsFromHTML', () => {
    const inputHTMLBody =`
<html>
    <body>
        <a href="/path/">
            Boot.dev Blog
        </a>
    </body>
</html>    
`
    const inputBaseURL ='https://blog.boot.dev/path/'
    const actual = getURLsFromHTML(inputHTMLBody,inputBaseURL)
    const expected=['https://blog.boot.dev/path/']
    expect(actual).toEqual(expected)
})
